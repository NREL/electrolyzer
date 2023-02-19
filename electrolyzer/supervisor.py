"""
This module defines the Hydrogen Electrolyzer control code.
"""
import numpy as np
import numpy.typing as npt
from attrs import field, define

from .stack import Stack
from .type_dec import NDArrayInt, NDArrayFloat, FromDictMixin


@define
class Supervisor(FromDictMixin):
    # Stack parameters #
    ####################

    dt: float
    stack: dict
    costs: dict  # TODO: should this be connected here?
    control: dict

    name: str = field(default="electrolyzer_001")
    description: str = field(default="A PEM electrolyzer model")

    control_type: str = field(init=False, default="BaselineDeg")

    # decision-based controller policies
    eager_on: bool = field(init=False, default=False)
    eager_off: bool = field(init=False, default=False)
    sequential: bool = field(init=False, default=False)
    even_dist: bool = field(init=False, default=False)
    baseline: bool = field(init=False, default=True)

    n_stacks: int = field(init=False, default=1)

    stack_min_power: float = field(init=False)
    stack_rating_kW: float = field(init=False)
    stack_rating: float = field(init=False)

    # Controller state #
    ####################

    # only for sequential controller TODO: find sneakier place to initialize this
    active_constant: NDArrayInt = field(init=False)

    # array of stack activation status 0 for inactive, 1 for active
    active: NDArrayInt = field(init=False)

    # array of stack waiting status 0 for active or inactive, 1 for waiting
    waiting: NDArrayInt = field(init=False)

    # again, only for sequential controller
    variable_stack: int = field(init=False, default=0)
    stack_rotation: NDArrayInt = field(init=False, default=[])
    stacks_on: int = field(init=False, default=0)
    stacks_waiting: int = field(init=False, default=0)
    stacks_off: NDArrayInt = field(init=False, default=[])
    stacks_waiting_vec: NDArrayInt = field(init=False)
    deg_state: NDArrayFloat = field(init=False)
    filter_width: int = field(init=False)
    past_power: NDArrayFloat = field(init=False)

    stacks: npt.NDArray = field(init=False)

    def __attrs_post_init__(self) -> None:
        """
        --- Current control_type Options ---

        Rotation-based electrolyzer action schemes:
            'PowerSharingRotation': power sharing, rotation
            'SequentialRotation': sequentially turn on electrolzyers, rotate
                electrolyzer roles based on set schedule (i.e. variable electrolyzer,
                etc.)

        Degredation-based electrolyzer action schemes:
            'EvenSplitEagerDeg': power sharing, eager to turn on electrolyzers
            'EvenSplitHesitantDeg': power sharing
            'SequentialEvenWearDeg': sequentially turn on electrolzyers, distribute
                wear evenly
            'SequentialSingleWearDeg': sequentially turn on electrolyzers, put all
                degradation on single electrolyzer
            'BaselineDeg': sequentially turn on and off electrolyzers but only when you
                have to

        Decision_ctrl controller action schemes:
            'BBBB': baseline controller - one big stack
            'HHDV': hesitant on, hesitant off, degradation order, variable power dist
            'HHDE': hesitant on, hesitant off, degradation order, even power dist
            'HHOV': hesitant on, hesitant off, rotation order, variable power dist
            'HHOE': hesitant on, hesitant off, rotation order, even power dist
            'HEDV': hesitant on, eager off, degradation order, variable power dist
            'HEDE': hesitant on, eager off, degradation order, even power dist
            'HEOV': hesitant on, eager off, rotation order, variable power dist
            'HEOE': hesitant on, eager off, rotation order, even power dist
            'EHDV': eager on, hesitant off, degradation order, variable power dist
            'EHDE': eager on, hesitant off, degradation order, even power dist
            'EHOV': eager on, hesitant off, rotation order, variable power dist
            'EHOE': eager on, hesitant off, rotation order, even power dist
        """
        self.control_type = self.control["control_type"]
        self.n_stacks = self.control["n_stacks"]

        if "sequential" in self.control_type.lower():
            # TODO: current filter width hardcoded at 5 min, make an input
            self.filter_width = round(300 / self.dt)

            # TODO: decide how to initialize past_power
            self.past_power = [0]

        if "Decision" in self.control_type:
            self.eager_on = self.control["policy"]["eager_on"]
            self.eager_off = self.control["policy"]["eager_off"]
            self.sequential = self.control["policy"]["sequential"]
            self.even_dist = self.control["policy"]["even_distribution"]
            self.baseline = self.control["policy"]["baseline"]

        self.active_constant = np.zeros(self.n_stacks)
        self.active = np.zeros(self.n_stacks)
        self.waiting = np.zeros(self.n_stacks)
        self.stacks_waiting_vec = np.zeros((self.n_stacks))
        self.deg_state = np.zeros(self.n_stacks)
        self.stacks = self.create_electrolyzer_stacks()

        # Query stack info from an initialized stack. All stacks have identical
        # ratings for now, but this may change in the future.
        self.stack_rating_kW = self.stacks[0].stack_rating_kW
        self.stack_rating = self.stacks[0].stack_rating
        self.stack_min_power = self.stacks[0].min_power

    # TODO: query stacks for on/off status instead of maintaining arrays

    def get_stacks_waiting(self):
        return [int(s.waiting) for s in self.stacks]

    def get_stacks_on(self):
        return [int(s.active) for s in self.stacks]

    def get_stacks_off(self):
        return [not int(s.active) for s in self.stacks]

    def create_electrolyzer_stacks(self):
        # initialize electrolyzer objects
        stacks = []
        self.stack["dt"] = self.dt
        for i in range(self.n_stacks):
            stacks.append(Stack.from_dict(self.stack))
            self.stack_rotation.append(i)
            print(
                "electrolyzer stack ",
                i + 1,
                "out of ",
                self.n_stacks,
                "has been initialized",
            )
        return stacks

    def update_stack_status(self):
        # Update stack status
        for i in range(self.n_stacks):
            if self.stacks[i].stack_on:
                self.stacks_on += 1
                self.active[i] = 1
            if self.stacks[i].stack_waiting:
                self.waiting[i] = 1
                self.stacks_waiting_vec[i] = 1
            else:
                self.waiting[i] = 0
                self.stacks_waiting_vec[i] = 0

    def run_control(self, power_in):
        """
        Inputs:
            power_in: power (W) to be consumed by the H2 farm every time step
        Returns:
            H2_mass_out: mass of h2 (kg) produced during each time step
            H2_mass_flow_rate: mfr of h2 (kg/s) during that time step
            power_left: power error (W) between what the stacks were
                supposed to consume and what they actually consumed
            curtailed_wind: power error (W) between available power_in and
                what the stacks are commanded to consume
        """

        # calculate stack power distribution
        if self.control_type == "PowerSharingRotation":
            stack_power, curtailed_wind = self.power_sharing_rotation(power_in)
        elif self.control_type == "SequentialRotation":
            stack_power, curtailed_wind = self.sequential_rotation(power_in)
        elif self.control_type == "EvenSplitEagerDeg":
            stack_power = self.distribute_power_equal_eager(power_in)
            curtailed_wind = 0
        elif self.control_type == "EvenSplitHesitantDeg":
            stack_power = self.distribute_power_equal_hesitant(power_in)
            curtailed_wind = 0
        elif self.control_type == "SequentialEvenWearDeg":
            stack_power = self.distribute_power_sequential_even_wear(power_in)
            curtailed_wind = 0
        elif self.control_type == "SequentialSingleWearDeg":
            stack_power = self.distribute_power_sequential_single_wear(power_in)
            curtailed_wind = 0
        elif self.control_type == "BaselineDeg":
            stack_power = self.baseline_controller(power_in)
            curtailed_wind = 0
        elif self.control_type == "Decision_ctrl":
            stack_power = self.decision_ctrl(power_in)
            curtailed_wind = 0

        # Query stacks for their status and turn them on or off as needed
        on_or_waiting = np.zeros(self.n_stacks)

        if ("deg" in self.control_type) or ("Decision" in self.control_type):
            for i in range(self.n_stacks):
                if self.stacks[i].stack_on or self.stacks[i].stack_waiting:
                    on_or_waiting[i] = 1

            # which stacks the controller thinks are on and which are actually on
            mismatch = self.active - on_or_waiting

            for i in range(len(mismatch)):
                # this means the controller wants an electrolyzer on and that
                # electrolyzer isnt on or waiting
                if mismatch[i] == 1:
                    self.stacks[i].turn_stack_on()
                elif mismatch[i] == 0:
                    pass
                # this means the controller wants an electrolyzer off and
                # the electrolyzer is on
                elif mismatch[i] == -1:
                    self.stacks[i].turn_stack_off()

            for i in range(self.n_stacks):
                if self.stacks[i].stack_waiting:
                    self.waiting[i] = 1
                else:
                    self.waiting[i] = 0

        active_actual = np.zeros(self.n_stacks)
        for i in range(self.n_stacks):
            if self.stacks[i].stack_on:
                active_actual[i] = 1

        power_left = 0
        H2_mass_out = 0
        self.stacks_waiting = 0
        self.stacks_on = 0
        H2_mass_flow_rate = np.zeros((self.n_stacks))

        # simulate 1 time step for each stack
        for i in range(self.n_stacks):
            H2_mfr, H2_mass_i, power_left_i = self.stacks[i].run(stack_power[i])

            self.deg_state[i] = self.stacks[i].V_degradation

            # Update stack status
            if self.stacks[i].stack_on:
                self.stacks_on += 1
                self.active[i] = 1
                on_or_waiting[i] = 1
            if self.stacks[i].stack_waiting:
                self.waiting[i] = 1
                self.stacks_waiting_vec[i] = 1
                on_or_waiting[i] = 1
            else:
                self.waiting[i] = 0
                self.stacks_waiting_vec[i] = 0

            H2_mass_flow_rate[i] = H2_mfr
            H2_mass_out += H2_mass_i
            power_left += power_left_i

        curtailed_wind = max(0, power_in - (np.dot(on_or_waiting, stack_power)))

        return H2_mass_out, H2_mass_flow_rate, power_left, curtailed_wind

    def power_sharing_rotation(self, power_in):
        # Control strategy that shares power between all electrolyzers equally
        if sum(self.active + self.waiting) == 0:
            P_indv = np.ones(1) * power_in / self.n_stacks
        else:
            P_indv = (
                np.ones(1) * power_in / sum(self.active + self.waiting)
            )  # divide the power evenely amongst electrolyzers
        P_indv_kW = P_indv / 1000

        stacks_supported = min(power_in // (self.stack_rating / 2), self.n_stacks)
        diff = int(stacks_supported - sum(self.active + self.waiting))

        # Power sharing control #
        #########################
        if diff > 0:
            # elif P_indv_kW > (0.8 * self.stack_rating_kW):
            if diff > 1 or P_indv_kW > (0.8 * self.stack_rating_kW):
                if sum(self.waiting) == 0 and sum(self.active) != self.n_stacks:
                    for i in range(0, diff):
                        ij = 0 + i
                        while self.active[self.stack_rotation[ij]] > 0:
                            ij += 1
                        self.turn_on_stack(self.stack_rotation[ij])

        if diff < 0:
            if P_indv_kW < (0.2 * self.stack_rating_kW):
                if sum(self.active) > 0:
                    self.turn_off_stack(self.stack_rotation[0])
                    self.stack_rotation = self.stack_rotation[1:] + [
                        self.stack_rotation[0]
                    ]

        new_stack_power = (
            np.ones((self.n_stacks)) * power_in / sum(self.active + self.waiting)
        )
        if (new_stack_power[0] / 1000) > (self.stack_rating_kW):
            curtailed_wind = (
                new_stack_power[0] - (self.stack_rating_kW * 1000)
            ) * self.stacks_on
            new_stack_power = np.ones((self.n_stacks)) * (self.stack_rating_kW * 1000)
        else:
            curtailed_wind = 0

        return new_stack_power, curtailed_wind

    def sequential_rotation(self, power_in):
        # Control strategy that fills up the electrolyzers sequentially

        P_indv = np.ones((self.n_stacks))

        p_in_kw = power_in / 1000

        n_full = p_in_kw // self.stack_rating_kW
        left_over_power = p_in_kw % self.stack_rating_kW
        stack_difference = int(n_full - sum(self.active + self.waiting))
        elec_var = self.stack_rotation[0]
        # other_elecs = [x for x in self.active if x not in self.stacks_off]

        # calculate the slope of power_in
        # (1) update past_power with current input
        if len(self.past_power) < self.filter_width:
            temp = np.zeros((len(self.past_power) + 1))
            temp[0:-1] = self.past_power[:]
            temp[-1] = np.copy(power_in)
            self.past_power = np.copy(temp)
        else:
            temp = np.zeros((len(self.past_power)))
            temp[0:-1] = self.past_power[1:]
            temp[-1] = np.copy(power_in)
            self.past_power = np.copy(temp)
        # (2) apply filter to find slope
        slope = power_in - np.mean(self.past_power)
        slope = np.mean(
            (np.mean(self.past_power[1:]) - np.mean(self.past_power[0:-1])) / self.dt
        )

        if stack_difference >= 0:
            # P_indv = P_indv * self.stack_rating_kW * 1000
            P_indv = P_indv * 0
            # for i in self.active:
            #     if i > 0:
            P_indv[self.active > 0] = self.stack_rating_kW * 1000
            P_indv[elec_var] = left_over_power * 1000
            curtailed_wind = (stack_difference * self.stack_rating_kW) + left_over_power
            if (
                sum(self.waiting) == 0
                and sum(self.active) != self.n_stacks
                and left_over_power > (0.15 * self.stack_rating_kW)
            ):
                for i in range(0, stack_difference):
                    ij = 0 + i
                    while self.active[self.stack_rotation[ij]] > 0:
                        ij += 1
                    self.turn_on_stack(self.stack_rotation[ij])
        if stack_difference < 0:
            curtailed_wind = 0
            P_indv = P_indv * 0

            P_indv[self.active > 0] = self.stack_rating_kW * 1000
            if stack_difference < -2:
                if sum(self.waiting) == 0 and sum(self.active) != self.n_stacks:
                    ij = 0
                    while self.active[self.stack_rotation[ij]] > 0:
                        ij += 1
                    off_stack = self.stack_rotation[ij]
                    # self.turn_on_stack(on_stack)
                    self.turn_off_stack(off_stack)
                    # P_indv[on_stack] = self.stack_rating_kW * 1000
                    P_indv[off_stack] = 0
                    P_indv[elec_var] = self.stack_rating_kW * 1000

            elif (
                (
                    left_over_power < (0.1 * self.stack_rating_kW)
                    and sum(self.waiting) == 0
                )
            ) or (stack_difference < -1 and sum(self.waiting) == 0):
                if sum(self.active) > 0 and slope < 0:
                    self.turn_off_stack(self.stack_rotation[0])
                    self.stack_rotation = self.stack_rotation[1:] + [
                        self.stack_rotation[0]
                    ]
                    elec_var = self.stack_rotation[0]
                    P_indv[elec_var] = self.stack_rating_kW * 1000
                    curtailed_wind = left_over_power
            elif (
                left_over_power < (0.1 * self.stack_rating_kW) and sum(self.waiting) > 0
            ):
                P_indv[self.waiting > 0] = (
                    (left_over_power + self.stack_rating_kW) * 1000 / 2
                )
                P_indv[elec_var] = (left_over_power + self.stack_rating_kW) * 1000 / 2
            # TODO : Find a way to turn on electrolyzers ahead of time based on wind
            # power signal slope
            elif left_over_power > (0.8 * self.stack_rating_kW) and slope > 0:
                # if or stack_difference
                if sum(self.waiting) == 0 and sum(self.active) != self.n_stacks:
                    ij = 0
                    while self.active[self.stack_rotation[ij]] > 0:
                        ij += 1
                    self.turn_on_stack(self.stack_rotation[ij])
                    # self.turn_on_stack()
                if sum(self.waiting) > 0:
                    P_indv[self.waiting > 0] = left_over_power * 1000 / 2
                    P_indv[elec_var] = left_over_power * 1000 / 2
                else:
                    P_indv[elec_var] = left_over_power * 1000
            else:
                if sum(self.waiting) > 0:
                    # if
                    P_indv[self.waiting > 0] = self.stack_rating_kW * 1000
                    P_indv[elec_var] = left_over_power * 1000
                else:
                    P_indv[elec_var] = left_over_power * 1000
        return P_indv, curtailed_wind * 1000

    def distribute_power_equal_eager(self, power_in):
        n_active = min([self.n_stacks, int(np.floor(power_in / self.stack_min_power))])
        if n_active > 0:
            # calculate curtailed wind here
            P_i = min([power_in / n_active, self.stack_rating])
            # curtailed_wind = max(0, (power_in/n_active) - self.stack_rating)
        else:
            P_i = 0

        if n_active == sum(self.active):
            pass  # do not need to turn on or off
        elif n_active > sum(self.active):
            diff = int(n_active - sum(self.active))
            self.active += self.get_healthiest_inactive(diff)
        elif n_active < sum(self.active):
            diff = int(sum(self.active) - n_active)
            self.active *= self.get_illest_active(diff)

        P_indv = P_i * self.active

        return P_indv

    def distribute_power_equal_hesitant(self, power_in):
        # dont turn on another electrolyzer until the other electrolyzers are all at
        # rated. turn off when all electrolyzers are below min power from previous step
        n_active = int(sum(self.active))

        # need to turn on another one
        if power_in > (self.stack_rating * n_active + self.stack_min_power):
            # number of stacks that need to be turned on
            diff = np.ceil(
                (power_in - n_active * self.stack_rating) / self.stack_rating
            )
            diff = min([diff, self.n_stacks - n_active])
            n_active += diff
        # not enough power to run all electrolyzers at minimum so we have to turn
        # some off
        elif power_in < self.stack_min_power * n_active:
            # want the number of elecs where np.ceil(P*min_rated)
            diff = n_active - np.floor(power_in / self.stack_min_power)
            n_active -= diff

        if n_active > 0:
            P_i = min([power_in / n_active, self.stack_rating])
        else:
            P_i = 0

        if n_active == sum(self.active):
            pass  # do not need to turn on or off
        elif n_active > sum(self.active):
            # need to turn on this many electrolzyers pick the healthiest elecs as
            # the next ones to turn on but only pick the healthiest from the ones
            # that are turned off
            diff = int(n_active - sum(self.active))
            self.active += self.get_healthiest_inactive(diff)

        elif n_active < sum(self.active):
            diff = int(sum(self.active) - n_active)  # need to turn off this many
            self.active *= self.get_illest_active(diff)

        P_indv = P_i * self.active

        return P_indv

    def distribute_power_sequential_even_wear(self, power_in):
        P_indv = np.zeros(self.n_stacks)

        n_active = np.min(
            [
                self.n_stacks,
                np.ceil((power_in - self.stack_min_power) / self.stack_rating),
            ]
        )

        diff = int(n_active - sum(self.active))

        if n_active > sum(self.active):
            stacks_to_turn_on = self.get_healthiest_inactive(diff)
            # for robustitude, this index should be [0][0] but it throws error so
            # come back to this
            self.variable_stack = np.nonzero(stacks_to_turn_on)[0][0]
            self.active += stacks_to_turn_on
            self.active_constant = np.copy(self.active)
            self.active_constant[self.variable_stack] = 0
        elif n_active < sum(self.active):
            self.active[self.variable_stack] = 0

            # need this to be get illest constant
            stacks_to_turn_off = self.get_illest_active(np.abs(diff))

            # these are the illest stacks from constant - the illest of these should
            # turn into variable
            self.variable_stack = np.nonzero(stacks_to_turn_off - 1)[0]

            # this only works if there is just one stack being turned off at at time
            self.active_constant[self.variable_stack] = 0

        variable_stack_P = np.min(
            [
                self.stack_rating,
                power_in - (sum(self.active_constant)) * self.stack_rating,
            ]
        )
        if variable_stack_P < self.stack_min_power:
            variable_stack_P = 0

        P_indv = self.stack_rating * self.active_constant
        P_indv[self.variable_stack] = variable_stack_P

        return P_indv

    def distribute_power_sequential_single_wear(self, power_in):
        # if we are trying a lot of different control strategies, maybe it would be
        # better to keep them in a seperate file and call them as a module
        P_indv = np.zeros(self.n_stacks)

        n_active = np.min(
            [
                self.n_stacks,
                np.ceil((power_in - self.stack_min_power) / self.stack_rating),
            ]
        )

        diff = int(n_active - sum(self.active))

        if n_active > sum(self.active):
            stacks_to_turn_on = self.get_healthiest_inactive(diff)

            # for robustitude, this index should be [0][0] but it throws error so
            # come back to this
            # self.variable_stack = np.nonzero(stacks_to_turn_on)[0][0]
            self.variable_stack = 0
            self.active += stacks_to_turn_on
            self.active_constant = np.copy(self.active)
            self.active_constant[self.variable_stack] = 0
        elif n_active < sum(self.active):
            # self.active[self.variable_stack] = 0

            # need this to be get illest constant
            stacks_to_turn_off = self.get_illest_active(np.abs(diff))

            # these are the illest stacks from constant - the illest of
            # these should turn into variabl
            # self.variable_stack = np.nonzero(stacks_to_turn_off-1)[0]

            # this only works if there is just one stack being turned off at at time
            # self.active_constant[self.variable_stack] = 0
            self.active_constant *= stacks_to_turn_off
            self.active *= stacks_to_turn_off

        variable_stack_P = np.min(
            [
                self.stack_rating,
                power_in - (sum(self.active_constant)) * self.stack_rating,
            ]
        )
        if variable_stack_P < self.stack_min_power:
            variable_stack_P = 0

        P_indv = self.stack_rating * self.active_constant
        P_indv[self.variable_stack] = variable_stack_P

        return P_indv

    def baseline_controller(self, power_in):
        """
        Hesitant to turn on, hesitant to turn off
        """

        p_avail = power_in

        # turn some on
        if (power_in > np.sum(self.active) * self.stack_rating) & (
            power_in > self.stack_min_power
        ):
            turn_on_ind = int(min([sum(self.active), self.n_stacks - 1]))
            self.stacks[turn_on_ind].turn_stack_on()
            self.active[turn_on_ind] = 1

        # turn some off
        elif power_in < np.sum(self.active) * self.stack_min_power:
            turn_off_ind = int(sum(self.active) - 1)
            self.stacks[turn_off_ind].turn_stack_off()
            self.active[turn_off_ind] = 0

        P_indv = self.stack_min_power * self.active
        p_avail -= sum(P_indv)

        for i in range(self.n_stacks):
            if p_avail >= (self.stack_rating - self.stack_min_power):
                P_indv[i] += self.stack_rating - self.stack_min_power
                p_avail -= self.stack_rating - self.stack_min_power
            elif p_avail < (self.stack_rating - self.stack_min_power):
                P_indv[i] += p_avail
                p_avail = 0

        return P_indv

    def decision_ctrl(self, power_in):
        num_on, num_off = self.check_turn_on_off(power_in)
        if num_on > 0:
            stacks_to_activate = self.get_next_stack_on(num_on)
            self.active += stacks_to_activate
        if num_off > 0:
            stacks_to_deactivate = self.get_next_stack_off(num_off)
            self.active *= stacks_to_deactivate

        P_i = self.distribute_power(power_in)
        return P_i

    def check_turn_on_off(self, P_avail):
        max_num_active = min(
            [self.n_stacks, int(np.floor(P_avail / self.stack_min_power))]
        )  # maximum possible number of electrolzyers running
        min_num_active = min(
            [
                self.n_stacks,
                int(
                    (P_avail > self.stack_min_power)
                    * np.ceil(P_avail / self.stack_rating)
                ),
            ]
        )  # minimum possible number of electrolyzers that can use all of P_avail

        n_active = sum(self.active)
        # self.maxmin.append([min_num_active, max_num_active, n_active])

        num_on = 0
        num_off = 0

        if self.eager_on & (not self.eager_off):
            # Option 1: Eager on, hesitant off
            if n_active > max_num_active:  # have the option       to turn off stacks
                num_on = 0
                num_off = max([0, n_active - max_num_active])
            elif n_active < max_num_active:  # have the option    to turn on stacks
                num_on = max([0, max_num_active - n_active])
                num_off = 0

        elif (not self.eager_on) & (not self.eager_off):
            # Option 2: Hesitant on, hesitant off
            if n_active > max_num_active:  # have the option       to turn off stacks
                num_on = 0
                num_off = max([0, n_active - max_num_active])
            elif n_active < min_num_active:  # have no choice but    to turn on stacks
                num_on = max([0, min_num_active - n_active])
                num_off = 0

        elif (not self.eager_on) & self.eager_off:
            # Option 3: Hesitant on, eager off
            if n_active > min_num_active:  # have no choice but     to turn off stacks
                num_on = 0
                num_off = max([0, n_active - min_num_active])
            elif n_active < min_num_active:  # have no choice but    to turn on stacks
                num_on = max([0, min_num_active - n_active])
                num_off = 0

            # Option 4: eager on, eager off has very frequent switching - not useful

        if self.baseline:
            if P_avail > self.n_stacks * self.stack_min_power:
                num_on = max([0, self.n_stacks - n_active])
                num_off = 0
            elif P_avail < self.n_stacks * self.stack_min_power:
                num_on = 0
                num_off = min([self.n_stacks, n_active])

        return num_on, num_off

    def get_next_stack_on(self, n_activate):
        if self.baseline:
            if n_activate > 0:
                stacks_to_activate = np.ones_like(self.active)
        else:
            if self.sequential:
                stacks_to_activate = np.zeros_like(self.active)
                activate_inds = np.arange(
                    sum(self.active), sum(self.active) + n_activate
                ).astype(int)
                for i in activate_inds:
                    stacks_to_activate[i] = 1
            else:
                stacks_to_activate = self.get_healthiest_inactive(n_activate)

        return stacks_to_activate

    def get_next_stack_off(self, n_deactivate):
        if self.baseline:
            if n_deactivate > 0:
                stacks_to_deactivate = np.zeros_like(self.active)
        else:
            if self.sequential:
                stacks_to_deactivate = np.ones_like(self.active)
                deactivate_inds = np.arange(
                    sum(self.active) - n_deactivate, sum(self.active)
                ).astype(int)
                for i in deactivate_inds:
                    stacks_to_deactivate[i] = 0
            else:
                stacks_to_deactivate = self.get_illest_active(n_deactivate)

        return stacks_to_deactivate

    def get_healthiest_inactive(self, n_activate):
        inactive = np.nonzero(self.active - 1)[0]
        ds = self.deg_state[inactive]
        deg_inds = np.argsort(ds)
        stacks_to_activate = np.zeros_like(self.active)
        stacks_to_activate[inactive[deg_inds[0 : int(n_activate)]]] = 1
        return stacks_to_activate

    def get_illest_active(self, n_deactivate):
        active_currently = np.nonzero(self.active)[0]
        ds = self.deg_state[active_currently]
        deg_inds = np.flip(np.argsort(ds))
        stacks_to_deactivate = np.ones_like(self.active)
        stacks_to_deactivate[active_currently[deg_inds[0 : int(n_deactivate)]]] = 0
        return stacks_to_deactivate

    def distribute_power(self, P_avail):
        P_i = np.zeros_like(self.active)

        for i, a in enumerate(self.active):
            if a:
                P_i[i] += self.stack_min_power
                P_avail -= self.stack_min_power

        if self.even_dist & (sum(self.active) > 0):
            P_indv = P_avail / sum(self.active)  # check this if power gets too large
            for i, a in enumerate(self.active):
                if a:
                    P_i[i] += P_indv
                    P_avail -= P_indv
        else:
            # permute
            """the way it is right now, the last electrolzyer in the active list will
            always be the variable one. If we want to use a different one then we can
            implement a permutation of the active array, run the forloop below then
            perform the inverse permutation on P_i afterwards"""

            for i, a in enumerate(self.active):
                if a:
                    if P_avail >= (self.stack_rating - self.stack_min_power):
                        P_i[i] += self.stack_rating - self.stack_min_power
                        P_avail -= self.stack_rating - self.stack_min_power
                    elif P_avail >= 0:
                        P_i[i] += P_avail
                        P_avail -= P_avail
            # unpermute

        return P_i

    def turn_off_stack(self, off_stack):
        # print('turn off stack')

        # off_stack = self.stack_rotation[0]
        # self.stacks_on.remove(off_stack)
        self.active[off_stack] = 0
        self.waiting[off_stack] = 0
        # self.stacks_off.append(off_stack)
        self.stacks[off_stack].turn_stack_off()
        # self.stacks_on = self.stacks_on - 1

    def turn_on_stack(self, on_stack):
        # print('turn on stack')

        # on_stack = self.stacks_off[0]
        # self.stacks_off.remove(on_stack)
        self.stacks[on_stack].turn_stack_on()
        self.waiting[on_stack] = 1
        # print(on_stack)

    # def get_healthiest_inactive(self, active, deg_state, n_activate):
    #     # TODO remove the arrays passed to this method since they are stored in
    #     # self already
    #     inact = np.nonzero(active - 1)[0]
    #     ds = deg_state[inact]
    #     deg_inds = np.argsort(ds)
    #     temp = np.zeros_like(active)
    #     temp[inact[deg_inds[0:n_activate]]] = 1
    #     return temp  # active + temp will turn on the stacks at 1s

    # def get_illest_active(self, active, deg_state, n_deactivate):
    #     act = np.nonzero(active)[0]
    #     ds = deg_state[act]
    #     deg_inds = np.flip(np.argsort(ds))
    #     temp = np.ones_like(active)
    #     temp[act[deg_inds[0:n_deactivate]]] = 0

    #     # active * temp will turn off the stacks at 0s
    #     return temp

    # want to be able to call this from time to time TODO
    # def calculater_LCOH_from_current_state()
    # roughly, find the slope of degradation over hydrogen then predict the lifetime
