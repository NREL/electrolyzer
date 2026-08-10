from enum import IntEnum
from pathlib import Path

import numpy as np
import openmdao.api as om

from electrolyzer.core.file_utils import load_yaml, make_unique_case_name
from electrolyzer.core.supported_models import supported_models
from electrolyzer.connectors.series_scalar import (
    CombineSerialComponents,  # , SplitAcrossSerialComponents
)
from electrolyzer.connectors.degradation_combiner import CombineDegradation
from electrolyzer.components.cell.cell_design_params import get_cell_params_for_model
from electrolyzer.components.classifiers.cell_classifier import CellClassification
from electrolyzer.components.classifiers.system_performance import SystemPerformance


class State(IntEnum):
    INITIALIZED = 0
    SETUP = 1
    RUN = 2
    POST_PROCESS = 3


class BERT:
    def __init__(self, config_input, make_n2=True, as_problem=True):
        self.create_n2 = make_n2
        self.supported_models = supported_models.copy()

        # read in config file; it's a yaml dict that looks like this:
        self.load_config(config_input)

        if as_problem:
            self.prob = om.Problem(reports=False)
            self.model = self.prob.model
            plant_group = om.Group()

            # Create the plant model group and add components
            self.plant = self.model.add_subsystem("plant", plant_group, promotes=["*"])
        else:
            self.plant = om.Group()

        self.create_controller()
        self.create_components()
        self.create_performance_aggregator()

        self.connect_system()

        if as_problem:
            self.create_recorder(self.prob)

        self.state = State.INITIALIZED

    def load_config(self, config_input):
        config = load_yaml(config_input)
        simulation_config = config.pop("simulation")
        system_config = config.pop("system")

        self.system_config = system_config
        self.plant_config = {"simulation": simulation_config}
        self.n_clusters = system_config["n_clusters"]
        self.control_var = system_config["control_variable"]
        if "cell" in config:
            # clusters have identical cell models
            self.identical_cells = True
        else:
            self.identical_cells = False
            msg = (
                "The ability to have clusters with different cell designs is not yet "
                "available. Please ensure your config has a ``cell`` section with "
                "the cell model and design parameters"
            )
            raise NotImplementedError(msg)

        self.config = config
        if self.control_var == "power":
            self.control_passed_var = "P"
        if self.control_var == "hydrogen":
            self.control_passed_var = "H2"

    def create_custom_models(self, model_config, config_parent_path, model_types, prefix=""):
        pass

    def setup(self):
        self.state = State.SETUP

        self.prob.setup()
        if self.create_n2:
            om.n2(self.prob, outfile=str(Path.cwd() / "n2_diagram.html"))
        self.prob.final_setup()
        self.prob.check_config(checks=["unconnected_inputs"], out_file=None)
        pass

    def run(self):
        if self.state < State.SETUP:
            self.setup()
        # self.prob.setup()
        # om.n2(self.prob, outfile=str(Path.cwd() / "n2_diagram.html"))
        # self.prob.final_setup()
        # self.prob.check_config(checks=["unconnected_inputs"], out_file=None)
        self.prob.run_model()
        self.state = State.RUN

    def post_process(self):
        pass

    def create_cluster_group(self):
        # Get the design parameters of the cell
        cell_design_params = get_cell_params_for_model(self.config["cell"].get("model", None))

        cluster_group = om.Group()

        # Step 2: Create controller cluster connector components
        pre_translator = self.create_controller_cluster_connector(cell_design_params)
        cluster_classifier = self.create_cluster_classification_component()
        # Translator has scale down + power to current conversion
        translator = self.create_controller_translator()
        # Step 3: Create the simulate block of a cluster
        simulator = self.create_cluster_simulation_block(cell_design_params)

        promotion_vars = [*cell_design_params, "I_min", "I_max"]
        cluster_group.add_subsystem("converter", pre_translator, promotes=promotion_vars)
        cluster_group.add_subsystem(
            "classifier", cluster_classifier, promotes=["I_min", "I_max", "n_cells", "n_stacks"]
        )
        cluster_group.add_subsystem("translator", translator, promotes=["n_stacks", "n_cells"])
        sim_prom_vars = [*promotion_vars, "n_stacks", "n_cells"]
        cluster_group.add_subsystem("simulation", simulator, promotes=sim_prom_vars)

        # simulation.dynamics gets I_min and I_max from the converter outputs
        # Connect the converter bounds to the
        cluster_group.connect(
            "converter.p2i.curve_coeffs", "translator.command_to_current.curve_coeffs"
        )
        cluster_group.connect("translator.command_to_current.I_command", "simulation.dynamics.I_in")

        # connect converter group stuff to classification block
        cluster_group.connect("converter.I_ref_points", "classifier.I_ref_points")
        cluster_group.connect("converter.ref_cell.J_out", "classifier.cell_classifier.J_in")
        for var in ["P", "H2", "O2", "V"]:
            cluster_group.connect(
                f"converter.ref_cell.{var}_cell_out", f"classifier.cell_classifier.{var}_in"
            )

        return cluster_group

    def create_controller(self):
        controller = self.create_controller_component()
        self.plant.add_subsystem("controller", controller)

    def create_performance_aggregator(self):
        ts_perf_mod = SystemPerformance(n_clusters=self.n_clusters)
        self.plant.add_subsystem("system_timeseries", ts_perf_mod)

        perf_mod = SystemPerformance(n_clusters=self.n_clusters)
        self.plant.add_subsystem("system_ub", perf_mod)

    def create_components(self):
        #

        # Get the design parameters of the cell
        cell_design_params = get_cell_params_for_model(self.config["cell"].get("model", None))

        # Step 1: Create cluster groups
        clusters = []
        # cluster_i = 0
        for cluster_i in range(self.n_clusters):
            cluster_comp = self.create_cluster_group()

            # NOTE: cell design params should only be promoted if all the clusters are identical
            if self.identical_cells:
                cluster_group = self.plant.add_subsystem(
                    f"Cluster{cluster_i}", cluster_comp, promotes=cell_design_params
                )
            else:
                cluster_group = self.plant.add_subsystem(f"Cluster{cluster_i}", cluster_comp)
            clusters.append(cluster_group)

        # Connect controller to cluster
        # self.plant.connect(
        #     f"controller.{self.control_passed_var}_command_{cluster_i}",
        #     f"Cluster{cluster_i}.translator.cluster_to_stack.{self.control_passed_var}_in"
        # )

        # cluster_group.connect("converter.I_ref_points", "classifier."
        # cluster_group.connect("converter.ref_cell.")

        self.clusters = clusters

    def connect_system(self):
        # Connect controller to cluster

        for cluster_i in range(0, self.n_clusters, 1):
            # Connect controller to cluster
            self.plant.connect(
                f"controller.{self.control_passed_var}_command_{cluster_i}",
                f"Cluster{cluster_i}.translator.cluster_to_stack.{self.control_passed_var}_in",
            )

        for cluster_i in range(0, self.n_clusters, 1):
            # connect the clusters to a system performance component
            # connect the classifier component and the simulation component
            for var in ["P", "H2", "O2", "V"]:
                self.plant.connect(
                    # part of scale_stack_to_cluster
                    f"Cluster{cluster_i}.simulation.Cluster_{var}",
                    f"system_timeseries.{var}_in_{cluster_i}",
                )
                self.plant.connect(
                    # part of classifier.stack_to_cluster_ub
                    f"Cluster{cluster_i}.classifier.{var}_max",
                    f"system_ub.{var}_in_{cluster_i}",
                )

    def create_cluster_simulation_block(self, cell_design_params):
        simulation = om.Group()

        cell_nom = self.create_cell_model()
        cell_real = self.create_cell_model()
        degradation = self.create_component("degradation")
        dynamics = self.create_component("dynamics")

        cell_scale_up = CombineSerialComponents(
            scaling_component="cells", n_comps=self.config["stack"]["n_cells"]
        )
        stack_scale_up = CombineSerialComponents(
            scaling_component="stacks", n_comps=self.config["cluster"]["n_stacks"]
        )
        degradation_combiner = CombineDegradation()

        simulation.add_subsystem("dynamics", dynamics, promotes=["I_min", "I_max"])
        simulation.add_subsystem("cell_nominal", cell_nom, promotes=cell_design_params)
        simulation.add_subsystem("degradation", degradation)
        simulation.add_subsystem("cell_real", cell_real, promotes=cell_design_params)
        simulation.add_subsystem("degradation_combiner", degradation_combiner)
        simulation.add_subsystem("scale_cell_to_stack", cell_scale_up, promotes_inputs=["n_cells"])

        scale_up_base_vars = ["J", "P", "H2", "O2", "V"]  # todo: add in I?
        cluster_out_prom_vars = [(f"{v}_out", f"Cluster_{v}") for v in scale_up_base_vars]
        simulation.add_subsystem(
            "scale_stack_to_cluster",
            stack_scale_up,
            promotes_inputs=["n_stacks"],
            promotes_outputs=cluster_out_prom_vars,
        )

        # connect dynamics current output to nominal cell current input
        simulation.connect("dynamics.I_out", "cell_nominal.I_in")
        # connect dynamics on/off status output to degradation on/off status input
        simulation.connect("dynamics.on_off_status", "degradation.on_off_status")
        # connect dynamics current output to the degradation nominal current input
        simulation.connect("dynamics.I_out", "degradation.I_in")
        # connect nominal cell voltage to the degradation
        simulation.connect("cell_nominal.V_cell_out", "degradation.V_cell_nominal")
        # connect the degraded current to the cell voltage
        simulation.connect("degradation.I_actual", "cell_real.I_in")

        # combine the degradation results at the cell level
        simulation.connect("cell_real.V_cell_out", "degradation_combiner.V_cell")
        simulation.connect("degradation.V_cell_degraded", "degradation_combiner.V_cell_deg")
        simulation.connect("degradation.I_actual", "degradation_combiner.I_actual")

        # connect the outputs from the real cell to the cell scale-up components
        simulation.connect("cell_real.J_out", "scale_cell_to_stack.J_in")
        simulation.connect("degradation.I_actual", "scale_cell_to_stack.I_in")
        # TODO: in the future, add a losses component and connect the outputs from that to the scale-up
        simulation.connect("cell_real.H2_cell_out", "scale_cell_to_stack.H2_in")
        simulation.connect("cell_real.O2_cell_out", "scale_cell_to_stack.O2_in")

        # connect the power and voltage from the degradation combiner to the scale-up components
        simulation.connect("degradation_combiner.V_cell_total", "scale_cell_to_stack.V_in")
        simulation.connect("degradation_combiner.P_cell_total", "scale_cell_to_stack.P_in")

        # Scale up from stack to cluster level

        for var in scale_up_base_vars:
            simulation.connect(f"scale_cell_to_stack.{var}_out", f"scale_stack_to_cluster.{var}_in")

        return simulation

    def create_controller_translator(self):
        translator = om.Group()

        cell_scale_down = self.create_scale_down_component(
            scale_comp="cells", n_comps=self.config["stack"]["n_cells"]
        )
        stack_scale_down = self.create_scale_down_component(
            scale_comp="stacks", n_comps=self.config["cluster"]["n_stacks"]
        )

        translator.add_subsystem(
            "cluster_to_stack",
            stack_scale_down,
            promotes_inputs=["n_stacks"],
        )
        translator.add_subsystem(
            "stack_to_cell",
            cell_scale_down,
            promotes_inputs=["n_cells"],
        )

        translator_comp = self.create_component(
            "control_command_converter", model_key="translator_model"
        )
        translator.add_subsystem("command_to_current", translator_comp)

        # Connect scale downs
        translator.connect("cluster_to_stack.P_out", "stack_to_cell.P_in")

        # Connect scale down power to current conversion
        translator.connect("stack_to_cell.P_out", "command_to_current.P_command")

        return translator

    def create_controller_cluster_connector(self, cell_design_params):
        """Group containing the:

        1. Bounds and reference point component (min/max bounds)
        2. Reference cell component
        3. Curve coefficient component

        Args:
            cell_design_params (list[str]): cell design parameters to promote

        Returns:
            om.Group: pre-simulation group
        """
        pre_converter_grp = om.Group()
        bounds_comp = self.create_bounds_component()

        # 1. Operational bounds and reference point component
        pre_converter_grp.add_subsystem(
            "IJ_ref",
            bounds_comp,
            promotes_inputs=["A_cell"],
            promotes_outputs=["I_ref_points", "I_min", "I_max"],
        )

        # 2. Reference cell model
        cell = self.create_cell_model()
        pre_converter_grp.add_subsystem("ref_cell", cell, promotes_inputs=cell_design_params)

        # 3. Curve coefficient component
        coeff_comp = self.create_component("control_command_converter", model_key="coeff_model")
        pre_converter_grp.add_subsystem("p2i", coeff_comp, promotes_inputs=["I_ref_points"])

        # Connect components

        # Connect the reference points to the cell
        pre_converter_grp.connect("I_ref_points", "ref_cell.I_in")
        # Connect the power output from the cell to the power to current thing
        pre_converter_grp.connect("ref_cell.P_cell_out", "p2i.P_ref_points")
        return pre_converter_grp

    def create_cluster_classification_component(self):
        # 4. Add the cell classification component to the system
        # The cell classification component inputs of J_in, P_in, H2_in, O2_in, V_in
        # and outputs min and max values of each input, plus min/max efficiency values
        # Cell model outputs P_cell_out, J_out, H2_cell_out, O2_cell_out, V_cell_out
        classifier_group = om.Group()

        cell_classifier = CellClassification(tech_config={}, plant_config=self.plant_config)
        classifier_group.add_subsystem(
            "cell_classifier",
            cell_classifier,
            promotes_inputs=["I_ref_points", "I_min", "I_max"],
            promotes_outputs=["efficiency_min", "efficiency_max"],
        )

        cell_scale_up_lb = CombineSerialComponents(
            scaling_component="cells", n_comps=self.config["stack"]["n_cells"]
        )
        stack_scale_up_lb = CombineSerialComponents(
            scaling_component="stacks", n_comps=self.config["cluster"]["n_stacks"]
        )

        cell_scale_up_ub = CombineSerialComponents(
            scaling_component="cells", n_comps=self.config["stack"]["n_cells"]
        )
        stack_scale_up_ub = CombineSerialComponents(
            scaling_component="stacks", n_comps=self.config["cluster"]["n_stacks"]
        )

        bounds_base_vars = ["J", "P", "H2", "O2", "V"]

        promoted_outputs_lb = [(f"{v}_out", f"{v}_min") for v in bounds_base_vars]
        promoted_outputs_ub = [(f"{v}_out", f"{v}_max") for v in bounds_base_vars]

        # # lower bounds
        classifier_group.add_subsystem(
            "cell_to_stack_lb", cell_scale_up_lb, promotes_inputs=["n_cells"]
        )
        classifier_group.add_subsystem(
            "stack_to_cluster_lb",
            stack_scale_up_lb,
            promotes_inputs=["n_stacks"],
            promotes_outputs=promoted_outputs_lb,
        )
        # # upper bounds
        classifier_group.add_subsystem(
            "cell_to_stack_ub", cell_scale_up_ub, promotes_inputs=["n_cells"]
        )
        classifier_group.add_subsystem(
            "stack_to_cluster_ub",
            stack_scale_up_ub,
            promotes_inputs=["n_stacks"],
            promotes_outputs=promoted_outputs_ub,
        )
        # Cluster0.classifier.cell_to_stack_lb.I_in
        # classifier_group.connect("I_min", "cell_to_stack_lb.I_in")
        for var in bounds_base_vars:
            #     # scale-up lower bounds
            classifier_group.connect(f"cell_classifier.{var}_min", f"cell_to_stack_lb.{var}_in")
            classifier_group.connect(f"cell_to_stack_lb.{var}_out", f"stack_to_cluster_lb.{var}_in")

            #     # scale-up upper bounds
            classifier_group.connect(f"cell_classifier.{var}_max", f"cell_to_stack_ub.{var}_in")
            classifier_group.connect(f"cell_to_stack_ub.{var}_out", f"stack_to_cluster_ub.{var}_in")

        return classifier_group

    def create_cell_model(self):
        cell_config = self.config["cell"]
        if (cell_model_name := cell_config.get("model", None)) is not None:
            if (cell_model := self.supported_models.get(cell_model_name, None)) is not None:
                return cell_model(plant_config=self.plant_config, tech_config=cell_config)
            raise ValueError(f"{cell_model_name} not found in supported models")
        raise ValueError("Missing model for ``cell`` component")

    def create_component(self, component_type: str, model_key="model"):
        config = self.config[component_type]
        if (model_name := config.get(model_key, None)) is not None:
            if (model := self.supported_models.get(model_name, None)) is not None:
                return model(plant_config=self.plant_config, tech_config=config)
            raise ValueError(
                f"{model_name} (specified as {component_type} model) not found in supported_models"
            )
            # TODO: Add checks on subbclass type for each component types
        raise ValueError(f"Missing model for ``{component_type}`` component")

    def create_bounds_component(self):
        component_type = "bounds"
        config = self.config["bounds"]
        cell_config = self.config["cell"].get("cell_parameters")
        if (model_name := config.get("model", None)) is not None:
            if (model := self.supported_models.get(model_name, None)) is not None:
                return model(cell_config=cell_config, tech_config=config)
            raise ValueError(
                f"{model_name} (specified as {component_type} model) not found in supported_models"
            )
            # TODO: Add checks on subbclass type for each component types
        raise ValueError(f"Missing model for ``{component_type}`` component")

    def create_scale_down_component(self, scale_comp: str, n_comps: int | float):
        if self.control_var == "power":
            model = self.supported_models["ScalePowerDown"]
            return model(scaling_component=scale_comp, n_components=n_comps)
        if self.control_var == "hydrogen":
            raise NotImplementedError("hydrogen is not yet a supported control variable")

    def create_controller_component(self):
        n_timesteps = int(self.plant_config["simulation"]["n_timesteps"])
        if "control_model" not in self.system_config:
            ivc_comp = om.IndepVarComp(
                name=f"{self.control_passed_var}_command_0",
                val=np.full(n_timesteps, 40.0),
                units="W",
            )
            if self.n_clusters > 1:
                msg = (
                    "Cannot run multiple clusters without a control model. "
                    "Please specify a control model"
                )
                raise NotImplementedError(msg)
            return ivc_comp
        controller_name = self.system_config["control_model"]
        controller_model = self.supported_models.get(controller_name)
        controller = controller_model(
            plant_config=self.plant_config,
            tech_config=self.system_config,
            n_clusters=self.n_clusters,
            control_variable=self.control_var,
        )
        return controller

    def create_recorder(self, opt_prob):
        # TODO: put this into pose_optimization one day

        if "recorder" not in self.config:
            return None

        folder_output = self.config.get("folder_output", Path.cwd())

        recorder_options = ["record_inputs", "record_outputs", "record_residuals"]
        if self.config["recorder"].get("flag", False):
            # Check that the output folder exists and create it if needed
            if not Path(folder_output).exists():
                Path.mkdir(folder_output, parents=True, exist_ok=True)

        if self.config["recorder"].get("flag", False):
            # Check that the output folder exists and create it if needed
            if not Path(folder_output).exists():
                Path.mkdir(folder_output, parents=True, exist_ok=True)

            overwrite_recorder = self.config["recorder"].get("overwrite_recorder", False)
            recorder_path = Path(folder_output) / self.config["recorder"]["file"]

            if not overwrite_recorder:
                # make a unique filename with the same base as self.config["recorder"]["file"]
                # separate out the filename without the extension
                file_base = self.config["recorder"]["file"].split(".sql")[0]

                recorder_fname = make_unique_case_name(
                    Path(folder_output), f"{file_base}.sql", ".sql"
                )
                recorder_path = Path(folder_output) / recorder_fname

            recorder_attachment = (
                self.config["recorder"].get("recorder_attachment", "driver").lower()
            )
            allowed_attachments = ["driver", "model"]
            if recorder_attachment not in allowed_attachments:
                msg = (
                    f"Invalid recorder attachment '{recorder_attachment}'. "
                    f"Currently supported options are {allowed_attachments}. "
                    "We recommend using 'driver' if running an optimization "
                    "or parameter sweep in parallel."
                )
                raise ValueError(msg)

            # Create recorder
            recorder = om.SqliteRecorder(recorder_path)

            if recorder_attachment == "model":
                # add the recorder to the model
                recorder_options += ["options_excludes"]

                opt_prob.model.add_recorder(recorder)

                for recorder_opt in recorder_options:
                    if recorder_opt in self.config["recorder"]:
                        opt_prob.model.recording_options[recorder_opt] = self.config[
                            "recorder"
                        ].get(recorder_opt)

                opt_prob.model.recording_options["includes"] = self.config["recorder"].get(
                    "includes", ["*"]
                )
                # opt_prob.model.recording_options["excludes"] = self.config["recorder"].get(
                #     "excludes", ["*resource_data"]
                # )
                return recorder_path

            if recorder_attachment == "driver":
                recorder_options += [
                    "record_constraints",
                    "record_derivative",
                    "record_desvars",
                    "record_objectives",
                ]
                # add the recorder to the driver
                opt_prob.driver.add_recorder(recorder)

                for recorder_opt in recorder_options:
                    if recorder_opt in self.config["recorder"]:
                        opt_prob.driver.recording_options[recorder_opt] = self.config[
                            "recorder"
                        ].get(recorder_opt)

                opt_prob.driver.recording_options["includes"] = self.config["recorder"].get(
                    "includes", ["*"]
                )
                # opt_prob.driver.recording_options["excludes"] = self.config["recorder"].get(
                #     "excludes", ["*resource_data"]
                # )
            return recorder_path

        return None
