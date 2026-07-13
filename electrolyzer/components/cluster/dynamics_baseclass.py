import openmdao.api as om


class DynamicsBase(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("plant_config", types=dict, default={})

    def setup(self):
        self.add_input("I_in", val=0.0, shape_by_conn=True, units="A")
        self.add_input("I_min", val=0.0, shape=1, units="A")
        self.add_input("I_max", val=0.0, shape=1, units="A")

        self.add_output("I_out", val=0.0, copy_shape="I_in", units="A")
        self.add_output("on_off_status", val=0.0, copy_shape="I_in", units="unitless")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
