import unittest
import main


class TestPipeline(unittest.TestCase):

    def test_extract_label(self):
        self.assertTrue(main.extract_label("1_1.jpg"))
        self.assertFalse(main.extract_label("93_0.jpg"))


if __name__ == '__main__':
    unittest.main()