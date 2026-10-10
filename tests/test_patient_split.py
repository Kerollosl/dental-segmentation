import unittest
from evaluate import validate_patient_manifest
class SplitTests(unittest.TestCase):
 def test_independent_patients(self):
  validate_patient_manifest([{'patient_id':str(i),'image':str(i)+'.png','split':s} for i,s in enumerate(['train','val','test'])])
 def test_leakage_rejected(self):
  with self.assertRaises(ValueError):validate_patient_manifest([{'patient_id':'a','image':str(i)+'.png','split':s} for i,s in enumerate(['train','val','test'])])
