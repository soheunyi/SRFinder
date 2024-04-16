class BlackBoxNetwork:
    def __init__(self):
        pass

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value):
        assert isinstance(value, int)
        self._input_dim = value

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        assert isinstance(value, int)
        self._output_dim = value
