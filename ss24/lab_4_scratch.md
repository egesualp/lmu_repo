### Example

#### Given Matrices and Vectors

- \( g^{(l-1)} \): \( 2 \times 1 \) vector
- \( W^{(l)} \): \( 2 \times 3 \) matrix
- \( g^{(l)} \): \( 3 \times 1 \) vector

#### Definitions

Let:
\[ W^{(l)} = \begin{pmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{pmatrix} \]

\[ g^{(l)} = \begin{pmatrix}
g_1 \\
g_2 \\
g_3
\end{pmatrix} \]

We want to calculate:
\[ g^{(l-1)} = W^{(l)} \cdot g^{(l)} \]

#### Matrix-Vector Multiplication

The resulting vector \( g^{(l-1)} \) will be:
\[ g^{(l-1)} = \begin{pmatrix}
w_{11} \cdot g_1 + w_{12} \cdot g_2 + w_{13} \cdot g_3 \\
w_{21} \cdot g_1 + w_{22} \cdot g_2 + w_{23} \cdot g_3
\end{pmatrix} \]

#### Example with Specific Numbers

Let's use specific numbers:
\[ W^{(l)} = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix} \]

\[ g^{(l)} = \begin{pmatrix}
7 \\
8 \\
9
\end{pmatrix} \]

Now, we calculate each element of \( g^{(l-1)} \):

1. First element:
\[ 1 \cdot 7 + 2 \cdot 8 + 3 \cdot 9 = 7 + 16 + 27 = 50 \]

2. Second element:
\[ 4 \cdot 7 + 5 \cdot 8 + 6 \cdot 9 = 28 + 40 + 54 = 122 \]

So, the resulting vector \( g^{(l-1)} \) is:
\[ g^{(l-1)} = \begin{pmatrix}
50 \\
122
\end{pmatrix} \]

### Summary

We started with:

- \( W^{(l)} \) as a \( 2 \times 3 \) matrix:
\[ \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{pmatrix} \]

- \( g^{(l)} \) as a \( 3 \times 1 \) vector:
\[ \begin{pmatrix}
7 \\
8 \\
9
\end{pmatrix} \]

After performing the matrix-vector multiplication \( W^{(l)} \cdot g^{(l)} \), we obtained \( g^{(l-1)} \) as a \( 2 \times 1 \) vector:
\[ \begin{pmatrix}
50 \\
122
\end{pmatrix} \]

This example illustrates how the dimensions and the actual multiplication process work in matrix-vector multiplication.
