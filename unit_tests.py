from SolutionFuncs import Cool_can, valCross
import unittest



class CoolCan_TestCase(unittest.TestCase):
    # Unit Tests for the CoolCan Function
    def test_AmbientTemp(self):
        # Provide a can temperature that is identical to the ambient temperature
        # Should output an average temp equal to the ambient
        t_start, t_end, T_avg, T_map = Cool_can(10, 0, 1000, 20, 0)
        self.assertAlmostEqual(T_avg[-1], 0, places=14)
    
    def test_InitialValue(self):
        # Tests to see if the first average temp of the can is equal to the initial
        # value provided to the IVP
        t_start, t_end, T_avg, T_map = Cool_can(10, 0, 1000, 20)
        self.assertAlmostEqual(T_avg[0], 35, places=14)


class valCross_TestCase(unittest.TestCase):
    # Unit Tests for the valCross Function
    def test_doesitknowifitcrossesthreshold(self):
        # Tests to see if it knows whether the threshold value is within the array
        testarray = [5, 4, 3, 2, 1]
        underthreshold = 0.2
        with self.assertRaises(ValueError):
            valCross(testarray, testarray, underthreshold)
    
    def test_doeslinearinterp(self):
        # tests to see if it does a proper linear interp between array vals
        indep = [4, 3, 2, 1, 0]
        depend = [5, 4, 3, 2, 1]
        self.assertAlmostEqual(valCross(indep, depend, 2.24), 3.24, places=12)
    
    def test_doesgrabindexval(self):
        # checks to see if it pulls an exact value if the threshold is in the array
        indep = [4, 3, 2, 1, 0]
        depend = [5, 4, 3, 2, 1]
        self.assertEqual(valCross(indep, depend, indep[3]), depend[3])

    def test_doescheckifdecreasing(self):
        # checks to see if it's recognizing the order of the array is decreasing in magnitude
        testarray = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            valCross(testarray, testarray, 2.4) 

if __name__ == '__main__':
    unittest.main()