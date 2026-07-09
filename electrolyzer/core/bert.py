import openmdao.api as om

from electrolyzer.core.file_utils import load_yaml
from electrolyzer.core.supported_models import supported_models


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
