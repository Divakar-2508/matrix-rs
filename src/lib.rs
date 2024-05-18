use std::ops::{Add, Index, IndexMut, Mul, Sub};

#[derive(Debug)]
pub enum MatrixError {
    InvalidColumnLen,
    DifferentDimensions,
    IndexOutOfBounds,
    ExpectedSquareMatrix,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    dim: Dimension,
    data: Vec<Vec<i32>>,
}


impl Matrix {
    pub fn new(row: usize, col: usize) -> Self {
        let data = vec![vec![0; col]; row];
        let dim = Dimension::new(row, col);
        Self { dim, data }
    }

    pub fn with_data<T, U>(data: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: IntoIterator<Item = i32>,
    {
        let data: Vec<Vec<i32>> = data
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        let dim = Dimension::new(data.len(), data[0].len());
        Self { dim, data }
    }

    pub fn new_with_dim(dim: Dimension) -> Self {
        let data = vec![vec![0; dim.col]; dim.row];
        Self { dim, data }
    }

    pub fn add_row(&mut self, data: &[i32]) -> Result<(), MatrixError> {
        if data.len() != self.dim.col {
            return Err(MatrixError::InvalidColumnLen);
        }
        self.data.push(data.to_vec());
        Ok(())
    }
    
    pub fn row_len(&self) -> usize {
        self.dim.row
    }

    pub fn col_len(&self) -> usize {
        self.dim.col
    }

    pub fn is_square_matrix(&self) -> bool {
        self.dim.row == self.dim.col
    }   

    pub fn diag(&self) -> Result<Vec<i32>, MatrixError> {
        if !self.is_square_matrix() {
            return Err(MatrixError::ExpectedSquareMatrix);
        }
        Ok((0..self.row_len()).map(|i| self[i][i]).collect())
    }

    pub fn counter_diag(&self) -> Result<Vec<i32>, MatrixError> {
        if !self.is_square_matrix() {
            return Err(MatrixError::ExpectedSquareMatrix);
        }
        let row_size = self.row_len();
        Ok((0..row_size).map(|i| self[i][row_size - i - 1]).collect())
    }

    pub fn sub_matrix(
        &self,
        row_start: usize,
        col_start: usize,
        row_end: usize,
        col_end: usize,
    ) -> Result<Self, MatrixError> {
        if row_start > self.row_len()
            || col_start > self.col_len()
            || row_end > self.row_len()
            || col_end > self.col_len()
        {
            return Err(MatrixError::IndexOutOfBounds);
        }

        let data: Vec<Vec<i32>> = (row_start..=row_end)
            .into_iter()
            .map(|row| self[row][col_start..=col_end].to_vec())
            .collect();

        Ok(Matrix::with_data(data))
    }

    pub fn add_matrix(&mut self, add_matrix: &Self) -> Result<(), MatrixError> {
        let col = add_matrix.col_len();
        let row = add_matrix.row_len();
        if self.dim != add_matrix.dim {
            return Err(MatrixError::DifferentDimensions);
        }

        for i in 0..row {
            for j in 0..col {
                self[i][j] += add_matrix[i][j];
            }
        }

        Ok(())
    }
}

impl Add for Matrix {
    type Output = Result<Self, MatrixError>;

    fn add(self, rhs: Self) -> Self::Output {
        add_matrices(&self, &rhs)
    }
}

impl Sub for Matrix {
    type Output = Result<Self, MatrixError>;

    fn sub(self, rhs: Self) -> Self::Output {
        subtract_matrices(&self, &rhs)
    }
}

impl Mul for Matrix {
    type Output = Result<Self, MatrixError>;

    fn mul(self, rhs: Self) -> Self::Output {
        product_matrices(&self, &rhs)
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<i32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.row_len() != other.row_len() || self.col_len() != other.col_len() {
            return false;
        }

        for i in 0..self.row_len() {
            for j in 0..self.col_len() {
                if self[i][j] != other[i][j] {
                    return false;
                }
            }
        }

        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Dimension {
    row: usize,
    col: usize,
}

impl Dimension {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

impl PartialEq for Dimension {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}

pub fn add_matrices(matrix1: &Matrix, matrix2: &Matrix) -> Result<Matrix, MatrixError> {
    let (row, col) = (matrix1.row_len(), matrix1.col_len());

    if row != matrix2.row_len() || col != matrix2.col_len() {
        return Err(MatrixError::DifferentDimensions);
    }

    let mut result_matrix = Matrix::new(row, col);

    for i in 0..row {
        for j in 0..col {
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    Ok(result_matrix)
}

pub fn product_matrices(matrix1: &Matrix, matrix2: &Matrix) -> Result<Matrix, MatrixError> {
    let (row1, col1) = (matrix1.row_len(), matrix1.col_len());
    let (row2, col2) = (matrix2.row_len(), matrix2.col_len());

    if col1 != row2 {
        return Err(MatrixError::DifferentDimensions);
    }

    let mut result_matrix = Matrix::new(row1, col2);

    for i in 0..row1 {
        for j in 0..col2 {
            for k in 0..col1 {
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][i];
            }
        }
    }

    Ok(result_matrix)
}

pub fn subtract_matrices(matrix1: &Matrix, matrix2: &Matrix) -> Result<Matrix, MatrixError> {
    if matrix1.dim != matrix2.dim {
        return Err(MatrixError::DifferentDimensions);
    }

    let mut result_matrix = Matrix::new_with_dim(matrix1.dim);

    for i in 0..matrix1.row_len() {
        for j in 0..matrix1.col_len() {
            result_matrix[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    Ok(result_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref MATRIX: Matrix = Matrix::with_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    }

    #[test]
    fn sub_matrix_test() {
        // let matrix = Matrix::with_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let sub_matrix = MATRIX.sub_matrix(0, 0, 1, 1);
        let mat2 = Matrix::with_data([[1, 2], [4, 5]]);
        assert_eq!(sub_matrix.unwrap(), mat2);
    }

    #[test]
    fn diag_elements() {
        assert_eq!(MATRIX.diag().unwrap(), vec![1, 5, 9]);
    }

    #[test]
    fn counter_diag_elements() {
        assert_eq!(MATRIX.counter_diag().unwrap(), vec![3, 5, 7]);
    }
}