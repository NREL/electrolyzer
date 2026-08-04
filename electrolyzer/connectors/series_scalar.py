import openmdao.api as om


class GenericSeriesConverter(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("scaling_component", types=str)
        self.options.declare("n_comps", types=(int, float), default=1.0)

    def setup(self):
        self.add_input(
            f"n_{self.options['scaling_component']}",
            val=self.options["n_comps"],
            shape=1,
            units="unitless",
        )
        vars_to_units = {
            "J": "A/(cm**2)",
            "I": "A",
            "P": "W",
            "H2": "kg/s",
            "O2": "kg/s",
            # "H2O": "kg/s",
            "V": "V",
            # "V_deg": "V"
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


class SplitAcrossSerialComponents(GenericSeriesConverter):
    """Scale power down"""

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")

            if o_name == "I_out" or o_name == "J_out":
                outputs[o_name] = inputs[in_name]
            else:
                outputs[o_name] = inputs[in_name] / inputs[f"n_{self.options['scaling_component']}"]


class CombineSerialComponents(GenericSeriesConverter):
    """Scale power down"""

    def setup(self):
        super().setup()

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")

            if o_name == "I_out" or o_name == "J_out":
                outputs[o_name] = inputs[in_name]
            else:
                outputs[o_name] = inputs[in_name] * inputs[f"n_{self.options['scaling_component']}"]
