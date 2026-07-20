from enum import IntEnum
from pathlib import Path

import numpy as np
import openmdao.api as om

from electrolyzer.core.file_utils import load_yaml
from electrolyzer.core.supported_models import supported_models


class State(IntEnum):
    INITIALIZED = 0
    SETUP = 1
    RUN = 2
    POST_PROCESS = 3


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

        self.create_controller()
        self.create_components()

        self.state = State.INITIALIZED

    def load_config(self, config_input):
        config = load_yaml(config_input)
        simulation_config = config.pop("simulation")
        system_config = config.pop("system")

        self.system_config = system_config
        self.plant_config = {"simulation": simulation_config}
        self.n_clusters = system_config["n_clusters"]
        self.control_var = system_config["control_variable"]
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

    def create_cluster_components(self):
        pass

    def create_controller(self):
        controller = self.create_controller_component()
        self.plant.add_subsystem("controller", controller)

    def create_components(self):
        #
        # Step 0: Create cluster groups
        clusters = []
        cluster_i = 0
        cluster_group = self.plant.add_subsystem(f"Cluster{cluster_i}", om.Group())
        clusters.append(cluster_group)

        # Step 1: Create controller cluster connector components
        pre_translator = self.create_controller_cluster_connector()
        # Translator has scale down + power to current conversion
        translator = self.create_controller_translator()
        # Step 2: Create the simulate block of a cluster
        simulator = self.create_cluster_simulation_block()

        cluster_group.add_subsystem(
            "converter", pre_translator, promotes=["A_cell", "I_min", "I_max"]
        )
        cluster_group.add_subsystem("translator", translator, promotes=["n_stacks", "n_cells"])
        cluster_group.add_subsystem("simulation", simulator, promotes=["I_min", "I_max", "A_cell"])

        # simulation.dynamics gets I_min and I_max from the converter outputs
        # Connect the converter bounds to the
        cluster_group.connect(
            "converter.p2i.curve_coeffs", "translator.command_to_current.curve_coeffs"
        )
        cluster_group.connect("translator.command_to_current.I_command", "simulation.dynamics.I_in")
        self.plant.connect(
            "controller.P_command", f"Cluster{cluster_i}.translator.cluster_to_stack.P_in"
        )

        self.clusters = clusters

    def create_cluster_simulation_block(self):
        simulation = om.Group()

        cell_nom = self.create_cell_model()
        cell_real = self.create_cell_model()
        degradation = self.create_component("degradation")
        dynamics = self.create_component("dynamics")

        simulation.add_subsystem("dynamics", dynamics, promotes=["I_min", "I_max"])
        simulation.add_subsystem("cell_nominal", cell_nom, promotes=["A_cell"])
        simulation.add_subsystem("degradation", degradation)
        simulation.add_subsystem("cell_real", cell_real, promotes=["A_cell"])

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

    def create_controller_cluster_connector(self):
        pre_converter_grp = om.Group()
        bounds_comp = self.create_bounds_component()
        pre_converter_grp.add_subsystem(
            "IJ_ref",
            bounds_comp,
            promotes_inputs=["A_cell"],
            promotes_outputs=["I_ref_points", "I_min", "I_max"],
        )

        cell = self.create_cell_model()
        pre_converter_grp.add_subsystem("ref_cell", cell, promotes_inputs=["A_cell"])

        coeff_comp = self.create_component("control_command_converter", model_key="coeff_model")
        pre_converter_grp.add_subsystem("p2i", coeff_comp, promotes_inputs=["I_ref_points"])

        # Connect the reference points to the cell
        pre_converter_grp.connect("I_ref_points", "ref_cell.I_in")
        # Connect the power output from the cell to the power to current thing
        pre_converter_grp.connect("ref_cell.P_cell_out", "p2i.P_ref_points")

        return pre_converter_grp

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
                name=f"{self.control_passed_var}_command", val=np.full(n_timesteps, 40.0), units="W"
            )
            return ivc_comp
        controller_name = self.system_config["control_model"]
        controller_model = self.supported_models(controller_name)
        controller = controller_model(
            plant_config=self.plant_config,
            tech_config=self.system_config,
            n_clusters=self.n_clusters,
            control_variable=self.control_var,
        )
        return controller
