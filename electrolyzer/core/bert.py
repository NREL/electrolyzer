import openmdao.api as om

from electrolyzer.core.file_utils import load_yaml
from electrolyzer.core.supported_models import supported_models
from electrolyzer.components.building_blocks import (
    IJBounds,
    ScaleDown,
    SimulateCell,
    CellDegradation,
    ClusterDynamics,
    CellPowerToCurrent,
)


class BERT:
    def __init__(self, config_input):
        self.supported_models = supported_models.copy()

        # read in config file; it's a yaml dict that looks like this:
        self.load_config(config_input)
        self.prob = om.Problem(reports=False)
        self.model = self.prob.model
        plant_group = om.Group()

        # Create the plant model group and add components
        self.plant = self.model.add_subsystem("plant", plant_group, promotes=["*"])

    def load_config(self, config_input):
        config = load_yaml(config_input)
        self.simulation_config = config["simulation"]
        self.electrolyzer_config = config["electrolyzer"]

    def create_custom_models(self, model_config, config_parent_path, model_types, prefix=""):
        pass

    def setup(self):
        pass

    def run(self):
        pass

    def post_process(self):
        pass

    def create_cluster_components(self):
        pass

    def create_components(self):
        #
        # Step 0: Create cluster groups
        clusters = []
        cluster_i = 0
        cluster_group = self.plant.add_subsystem(f"Cluster{cluster_i}", om.Group())
        clusters.append(cluster_group)

        # Step 1: Create controller cluster connector components
        self.create_controller_cluster_connector(cluster_group)
        # Step 2: Create the simulate block of a cluster
        self.create_cluster_simulation_block(cluster_group)

        self.clusters = clusters

    def connect_compnents(self):
        pass
        # NOTE: see if we can connect things within functions
        # Step 1: Connecter parts within the controller cluster_connector
        # self.connect_controller_cluster_connector(self.clusters[cluster_i])

    def create_cluster_simulation_block(self, cluster_group):
        # TODO: replace the simulation group w/o requiring the cluster group input
        # TODO: add to cluster group in method that calls this one
        simulation = cluster_group.add_subsystem(
            "simulation", om.Group(), promotes=["I_min", "I_max", "A_cell"]
        )

        simulation.add_subsystem("dynamics", ClusterDynamics(), promotes=["I_min", "I_max"])
        simulation.add_subsystem("cell_nominal", SimulateCell(), promotes=["A_cell"])
        simulation.add_subsystem("degradation", CellDegradation())
        simulation.add_subsystem("cell_real", SimulateCell(), promotes=["A_cell"])

        # connect dynamics current output to nominal cell current input
        simulation.connect("dynamics.I_out", "cell_nominal.I_in")
        # connect dynamics on/off status output to degradation on/off status input
        simulation.connect("dynamics.on_off_status", "degradation.on_off_status")
        # connect dynamics current output to the degradation nominal current input
        simulation.connect("dynamics.I_out", "degradation.I_in")
        # connect nominal cell voltage to the degradation
        simulation.connect("cell_nominal.V_cell_out", "degradation.V_cell_nominal")
        # connect the degraded current to the cell voltage
        simulation.connect("degradation.I_out", "cell_real.I_in")

    def create_controller_cluster_connector(self, cluster_group):
        # "Pre-processing", connects cluster to controller

        cell_scale_down = ScaleDown(scaling_component="cells")
        stack_scale_down = ScaleDown(scaling_component="stacks")

        scale_down = cluster_group.add_subsystem(
            "scale_down", om.Group(), promotes=["n_stacks", "n_cells"]
        )
        scale_down.add_subsystem(
            "cluster_to_stack",
            stack_scale_down,
            promotes_inputs=["n_stacks"],  # , ("P_in", "P_cluster_in")],
            # promotes_outputs=[("P_out", "P_stack_in")],
        )
        scale_down.add_subsystem(
            "stack_to_cell",
            cell_scale_down,
            promotes_inputs=["n_cells"],  # , ("P_in", "P_stack_in")],
            # promotes_outputs=[("P_out", "P_cell_in")],
        )

        # bounds_component = IJBounds()
        # cell_component = SimulateCell()
        # current_translator_comp = CellPowerToCurrent()
        pre_converter_grp = cluster_group.add_subsystem(
            "converter", om.Group(), promotes=["A_cell", "I_min", "I_max"]
        )
        pre_converter_grp.add_subsystem(
            "IJ_ref",
            IJBounds(),
            promotes_inputs=["A_cell"],
            promotes_outputs=["I_ref_points", "I_min", "I_max"],
        )
        pre_converter_grp.add_subsystem("ref_cell", SimulateCell(), promotes_inputs=["A_cell"])
        pre_converter_grp.add_subsystem(
            "p2i", CellPowerToCurrent(), promotes_inputs=["I_ref_points"]
        )

        # def connect_controller_cluster_connector(self, cluster_group):
        # Connect scale downs
        cluster_group.connect("scale_down.cluster_to_stack.P_out", "scale_down.stack_to_cell.P_in")
        # Connect the reference points to the cell
        cluster_group.connect("converter.I_ref_points", "converter.ref_cell.I_in")
        # Connect the power output from the cell to the power to current thing
        cluster_group.connect("converter.ref_cell.P_cell_out", "converter.p2i.P_ref_points")
        # Connect scale down power to current conversion
        cluster_group.connect("scale_down.stack_to_cell.P_out", "converter.p2i.P_command")

    # TODO: Connect the power from the "controller" to the CellPowerToCurrent
    # cluster_group.connect("ivc.P_command", "scale_down.P_cluster_in")

    # def create_preprocessing_converter(self, control_cmd_type):
    #     if control_cmd_type == "power":

    # def create_technologies(self, control_coverter_signal):
    #     cell_model_name = self.electrolyzer_config["cell_model"]["model"]
    #     if control_coverter_signal == "hydrogen":
    #         self.supported_models[cell_model_name](
    #             plant_config=self.simulation_config,
    #             tech_config=self.electrolyzer_config,
    #             mode="h2_dmd_adj",
    #         )
    #     else:
    #         self.supported_models[cell_model_name](
    #             plant_config=self.simulation_config,
    #             tech_config=self.electrolyzer_config,
    #             mode="normal",
    #         )
    #     tech_group = self.plant.add_subsystem(tech_name, om.Group())
    #     tech_group.add_subsystem(name, tech_object, promots=["*"])

    # def create_electrolyzer_system(self):
    #     # make each thing a cluster
    #     self.plant.add_subsystem(tech_name, om.Group())
    #     pass

    def connect_components(self):
        # stack needs inputs of nominal current, status, and nominal cell voltage
        # stack outputs actual current, degradation voltage
        # hydrogen production is a function of current and current density

        # connect current from cluster to stack
        self.plant.connect("cluster.current_out", "stack.current_in")
        # connect cluster status to stack
        self.plant.connect("cluster.status_out", "stack.status_in")
        # connect stack current to cell
        self.plant.connect("stack.current_out", "cell.current_in")
        # connect cell voltage to stack
        self.plant.connect("cell.cell_voltage", "stack.voltage_in")

        # Cluster -> Stack: nominal current input
        # Cluster -> Cell: nominal current input
        # Cell -> Stack: nominal cell voltage
        # Cluster -> Stack: Cluster on/off status
        # Stack -> Cell: actual current (degradation adjusted)

        # Cell -> Stack: Stack needs nominal current input and nominal cell voltage input
        # Stack outputs actual current to cell

        pass
