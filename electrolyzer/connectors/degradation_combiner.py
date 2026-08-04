import openmdao.api as om


class CombineDegradation(om.ExplicitComponent):
    """Combine cell voltage with the cell degradation"""

    def initialize(self):
        pass

    def setup(self):
        self.add_input("V_cell_deg", val=0.0, shape_by_conn=True, units="V")
        self.add_input("V_cell", val=0.0, copy_shape="V_cell_deg", units="V")
        self.add_input("I_actual", val=0.0, copy_shape="V_cell_deg", units="A")

        self.add_output("V_cell_total", val=0.0, copy_shape="V_cell_deg", units="V")
        self.add_output("P_cell_total", val=0.0, copy_shape="V_cell_deg", units="W")

    def compute(self, inputs, outputs):
        outputs["V_cell_total"] = inputs["V_cell_deg"] + inputs["V_cell"]
        outputs["P_cell_total"] = inputs["I_actual"] * outputs["V_cell_total"]
