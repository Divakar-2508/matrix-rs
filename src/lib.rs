#[allow(dead_code)]
use std::ops::{Index, IndexMut};

pub struct Matrix {
    row: usize,
    col: usize,
    data: Vec<Vec<i32>>,
}

enum MatrixError {
    InvalidColumnLen,
    DifferentDimensions,
}

impl Matrix {
    fn new(row: usize, col: usize) -> Self {
        let data = vec![vec![0; col]; row];
        Self { row, col, data }
    }

    fn add_row(&mut self, data: &[i32]) -> Result<(), MatrixError> {
        if data.len() != self.col {
            return Err(MatrixError::InvalidColumnLen);
        }
        self.data.push(data.to_vec());
        Ok(())
    }

    fn with_data(data: &[&[i32]]) -> Self {
        let data: Vec<Vec<i32>> = data.into_iter().map(|row| row.to_vec()).collect();
        let row = data.len();
        let col = data[0].len();
        Self { row, col, data }
    }

    fn row(&self) -> usize {
        self.row
    }

    fn col(&self) -> usize {
        self.col
    }

    fn add_matrix(&mut self, add_matrix: &Self) -> Result<(), MatrixError> {
        let col = add_matrix.col();
        let row = add_matrix.row();
        if self.col != col || self.row != row {
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

fn add_matrices(matrix1: &Matrix, matrix2: &Matrix) -> Result<Matrix, MatrixError> {
    let (row, col) = (matrix1.row(), matrix1.col());

    if row != matrix2.row() || col != matrix2.col() {
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

fn product_matrices(matrix1: &Matrix, matrix2: &Matrix) -> Result<Matrix, MatrixError> {
    let (row1, col1) = (matrix1.row(), matrix1.col());
    let (row2, col2) = (matrix2.row(), matrix2.col());

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
