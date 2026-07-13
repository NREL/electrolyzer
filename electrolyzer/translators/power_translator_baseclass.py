import openmdao.api as om


class PowerToCurrentBase(om.ExplicitComponent):
    # TODO: REMOVE OR REFACTOR
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("plant_config", types=dict, default={})

    def setup(self):
        self.add_input("I_ref_points", val=0.0, shape_by_conn=True, units="A")
        # self.add_input("J_ref_points", val=0.0, copy_shape="I_ref_points", units="A/(cm**2)")
        self.add_input("P_ref_points", val=0.0, copy_shape="I_ref_points", units="W")
        # NOTE: could use V_ref_points instead and calculate power in compute()

        self.add_input("P_command", val=0.0, shape_by_conn=True, units="W")
        self.add_output("I_command", val=0.0, copy_shape="P_command", units="A")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
