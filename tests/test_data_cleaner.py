import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que el método drop_invalid_rows elimina correctamente las filas
        que contienen valores faltantes (NaN o None) en las columnas especificadas.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        result = cleaner.drop_invalid_rows(df, ["name", "age"])

        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)

        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que el método drop_invalid_rows lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica que el método trim_strings elimina correctamente los espacios
        en blanco sin modificar el DataFrame original.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        original_df = df.copy(deep=True)

        result = cleaner.trim_strings(df, ["name"])

        # Verificar que el DataFrame original no cambió
        self.assertEqual(df.loc[0, "name"], " Alice ")
        self.assertEqual(df.loc[3, "name"], " Carol  ")

        # Verificar que el resultado sí fue limpiado
        self.assertEqual(result.loc[0, "name"], "Alice")
        self.assertEqual(result.loc[3, "name"], "Carol")

        # Verificar que columnas no especificadas permanecen iguales
        pdt.assert_series_equal(result["city"], original_df["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que el método trim_strings lanza un TypeError cuando
        se llama con una columna numérica.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica que remove_outliers_iqr elimina correctamente outliers.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)

        self.assertNotIn(120, result["age"].values)
        self.assertIn(25, result["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que remove_outliers_iqr lanza un KeyError cuando
        la columna no existe.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que remove_outliers_iqr lanza un TypeError cuando
        la columna no es numérica.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")

if __name__ == "__main__":
    unittest.main()
