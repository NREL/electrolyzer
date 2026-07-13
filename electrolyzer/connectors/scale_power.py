import openmdao.api as om


class ScalePowerBase(om.ExplicitComponent):
    """Scale things down"""

    def initialize(self):
        self.options.declare("scaling_component", types=str)

    def setup(self):
        self.add_input(f"n_{self.options['scaling_component']}", val=1.0, shape=1, units="unitless")
        vars_to_units = {
            # "I": "A",
            "P": "W",
            # "H2": "kg/s",
            # "V_stack": "V",
            # "V_deg_stack": "V"
        }

        ref_shape = None
        for v, u in vars_to_units.items():
            if ref_shape is None:
                self.add_input(f"{v}_in", val=0.0, shape_by_conn=True, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=f"{v}_in", units=u)
                ref_shape = f"{v}_in"
            else:
                self.add_input(f"{v}_in", val=0.0, copy_shape=ref_shape, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=ref_shape, units=u)


class ScalePowerDown(ScalePowerBase):
    """Scale power down"""

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")
            outputs[o_name] = inputs[in_name] / inputs[f"n_{self.options['scaling_component']}"]


class ScalePowerUp(ScalePowerBase):
    """Scale power down"""

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")
            outputs[o_name] = inputs[in_name] * inputs[f"n_{self.options['scaling_component']}"]
