use std::ops::{Add, Mul, Sub};
#[allow(dead_code)]
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Matrix {
    dim: Dimension,
    data: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Dimension {
    row: usize,
    col: usize,
}

impl Dimension {
    fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
    fn get_row(&self) -> usize {
        self.row
    }

    fn get_col(&self) -> usize {
        self.col
    }
}

impl PartialEq for Dimension {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}

#[derive(Debug)]
pub enum MatrixError {
    InvalidColumnLen,
    DifferentDimensions,
    IndexOutOfBounds,
}

impl Matrix {
    fn new(row: usize, col: usize) -> Self {
        let data = vec![vec![0; col]; row];
        let dim = Dimension::new(row, col);
        Self { dim, data }
    }
    fn with_data<T, U>(data: T) -> Self
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

    fn new_with_dim(dim: Dimension) -> Self {
        let data = vec![vec![0; dim.col]; dim.row];
        Self { dim, data }
    }

    fn add_row(&mut self, data: &[i32]) -> Result<(), MatrixError> {
        if data.len() != self.dim.col {
            return Err(MatrixError::InvalidColumnLen);
        }
        self.data.push(data.to_vec());
        Ok(())
    }

    fn is_square_matrix(&self) -> bool {
        self.dim.row == self.dim.col
    }

    fn sub_matrix(
        &self,
        row_start: usize,
        col_start: usize,
        row_end: usize,
        col_end: usize,
    ) -> Result<Self, MatrixError> {
        if row_start > self.row()
            || col_start > self.col()
            || row_end > self.row()
            || col_end > self.col()
        {
            return Err(MatrixError::IndexOutOfBounds);
        }

        let data: Vec<Vec<i32>> = (row_start..=row_end)
            .into_iter()
            .map(|row| self[row][col_start..=col_end].to_vec())
            .collect();

        Ok(Matrix::with_data(data))
    }

    // fn determinant(&self) -> Option<usize> {
    //     todo!();
    //     if !self.is_square_matrix() {
    //         return None;
    //     }
    //
    //     if self.row() == 2 {
    //         return Ok(self[0][0] * self[1][1] - self[0][1] * self[1][0]);
    //     }
    //
    //     let mut det = 0;
    //
    //     for i in 0 .. self.col() {
    //         let sub_det = self[0][i] * self.sub_matrix(row_start, col_start, row_end, col_end)
    //         if i % 2 == 0 {
    //
    //         } else {
    //
    //         }
    //     }
    // }

    fn row(&self) -> usize {
        self.dim.row
    }

    fn col(&self) -> usize {
        self.dim.col
    }

    fn add_matrix(&mut self, add_matrix: &Self) -> Result<(), MatrixError> {
        let col = add_matrix.col();
        let row = add_matrix.row();
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

fn subtract_matrices(matrix1: &Matrix, matrix2: &Matrix) -> Result<Matrix, MatrixError> {
    if matrix1.dim != matrix2.dim {
        return Err(MatrixError::DifferentDimensions);
    }

    let mut result_matrix = Matrix::new_with_dim(matrix1.dim);

    for i in 0..matrix1.row() {
        for j in 0..matrix1.col() {
            result_matrix[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    Ok(result_matrix)
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
        if self.row() != other.row() || self.col() != other.col() {
            return false;
        }

        for i in 0..self.row() {
            for j in 0..self.col() {
                if self[i][j] != other[i][j] {
                    return false;
                }
            }
        }

        true
    }
}

#[test]
fn sub_matrix_test() {
    let matrix = Matrix::with_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let sub_matrix = matrix.sub_matrix(0, 0, 1, 1);
    let mat2 = Matrix::with_data([[1, 2], [4, 5]]);
    assert_eq!(sub_matrix.unwrap(), mat2);
}
