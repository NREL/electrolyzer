import numpy as np
import openmdao.api as om


class SystemPerformance(om.ExplicitComponent):
    """Connect clusters"""

    def initialize(self):
        self.options.declare("n_clusters", types=(int, float), default=1.0)

    def setup(self):
        self.vars_to_units = {
            # "J": "A/(cm**2)",
            # "I": "A",
            "P": "W",
            "H2": "kg/s",
            "O2": "kg/s",
            # "H2O": "kg/s",
            "V": "V",
            # "V_deg": "V"
        }

        ref_shape = None
        for v, u in self.vars_to_units.items():
            for i in range(0, int(self.options["n_clusters"])):
                if ref_shape is None:
                    self.add_input(f"{v}_in_{i}", val=0.0, shape_by_conn=True, units=u)
                    ref_shape = f"{v}_in_{i}"
                else:
                    self.add_input(f"{v}_in_{i}", val=0.0, copy_shape=ref_shape, units=u)
            self.add_output(f"{v}_out", val=0.0, copy_shape=ref_shape, units=u)

        self.ref_shape = ref_shape

    def compute(self, inputs, outputs):
        n_timesteps = len(inputs[self.ref_shape])

        for v in self.vars_to_units.keys():
            var_cnt = np.zeros(n_timesteps)
            for i in range(0, int(self.options["n_clusters"])):
                var_cnt += inputs[f"{v}_in_{i}"]
            outputs[f"{v}_out"] = var_cnt
