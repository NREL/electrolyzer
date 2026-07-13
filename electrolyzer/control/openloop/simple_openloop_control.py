import openmdao.api as om
from attrs import field, define

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import contains


@define(kw_only=True)
class OLControlConfig(BaseConfig):
    # n_clusters: int = field(converter=int, validator=validators.gt(0.0))
    control_cmd: str = field(
        converter=(str.lower, str.strip), validator=contains(["power", "hydrogen"])
    )


class OLBasicSplit(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict, default={})
        self.options.declare("tech_config", types=dict)
        self.options.declare("n_clusters", types=int)

    def setup(self):
        # self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        # self.dt = self.options["plant_config"]["simulation"]["dt"]
        self.config = OLControlConfig.from_dict(self.options["tech_config"]["control_parameters"])
        self.n_clusters = self.options["n_clusters"]

        if self.config.control_cmd == "power":
            # output_cmd_fmt = "power_cmd_{ci}"
            self.add_input("P_command", val=0.0, shape_by_conn=True, units="kW")
            for ci in range(self.n_clusters):
                self.add_output(f"P_command_{ci}", val=0.0, copy_shape="P_command", units="kW")
        else:
            # output_cmd_fmt = "hydrogen_cmd_{ci}"
            self.add_input("H2_command", val=0.0, shape_by_conn=True, units="kg/h")
            for ci in range(self.n_clusters):
                self.add_output(f"H2_command_{ci}", val=0.0, copy_shape="H2_command", units="kg/h")

        # design variables
        # self.add_input("n_clusters", val=self.config.n_clusters, units="unitless")

    def compute(self, inputs, outputs):
        if self.config.control_cmd == "power":
            power_per_cluster = inputs["P_command"] / self.config.n_clusters
            for ci in range(self.n_clusters):
                outputs[f"P_command_{ci}"] = power_per_cluster
        else:
            h2_per_cluster = inputs["H2_command"] / self.config.n_clusters
            for ci in range(self.n_clusters):
                outputs[f"H2_command_{ci}"] = h2_per_cluster
