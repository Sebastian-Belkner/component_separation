# %%
import unittest
import component_separation.powspec as pows
import logging
from cs_util import Planckf, Plancks

logging.basicConfig(format='   %(levelname)s:      %(message)s', level=logging.DEBUG)

freqfilter = [
    Planckf.LFI_1.value,
    Planckf.LFI_2.value,
    Planckf.HFI_1.value,
    Planckf.HFI_5.value,
    Planckf.HFI_6.value]
    
specfilter = ["TE", "TB", "ET", "BT"]

class TestMethods(unittest.TestCase):
    def test1(self):
        #TODO think of a good test
        self.assertEqual(len(pows.get_data(freqfilter)), 3)

    def test2(self):
        #TODO think of a good test
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test3(self):
        #TODO think of a good test
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

# %%
if __name__ == '__main__':
    unittest.main()