

from quantem.io.serialization import AutoSerialize


# base class for quantem datasets
class Dataset(AutoSerialize):
    def __init__(
    	self,
    ):
        self.data = np.random.rand(100, 100)
        self.info = {"key": "value"}
        # self.child =
        self.child.info = {"key": "value"}
     
    
    def test_sum(self):
        return self.data.sum()




