import unittest
from class_mapping import mappings,validate_model_names
class MappingTests(unittest.TestCase):
 def test_all_teeth_and_conditions(self):
  fdi,universal=mappings()
  self.assertEqual(set(universal[i] for i in range(32)),set(map(str,range(1,33))))
  self.assertEqual(len(set(fdi[i] for i in range(32))),32)
  self.assertEqual(len(fdi),42)
  validate_model_names(fdi)
 def test_drift_rejected(self):
  fdi,_=mappings();fdi[0]="wrong"
  with self.assertRaises(ValueError):validate_model_names(fdi)
