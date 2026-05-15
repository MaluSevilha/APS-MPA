# Backpropagation - Autoencoder 3-2-3

library(tidyverse)
library(plotly)

# identity activation (equivalente a fazer uma PCA)
# sig  <- function(u) u
# dsig <- function(u) 1

# logistic activation
sig  <- function(u) 1 / (1 + exp(-u))
dsig <- function(u) sig(u) * (1 - sig(u))

generate_3d_data <- function(m, w1 = 0.1, w2 = 0.3, noise = 0.1) {
    angles <- runif(m) * 3 * (pi / 2) - 0.5
    data <- matrix(0, nrow = m, ncol = 3)
    data[, 1] <- cos(angles) + sin(angles) / 2 + noise * rnorm(m) / 2
    data[, 2] <- sin(angles) * 0.7 + noise * rnorm(m) / 2
    data[, 3] <- data[, 1] * w1 + data[, 2] * w2 + noise * rnorm(m)
    colnames(data) <- c("x1", "x2", "x3")
    data
}

set.seed(42)

X <- scale(generate_3d_data(100), center = TRUE, scale = FALSE)

v11 <- rnorm(1); v12 <- rnorm(1); v13 <- rnorm(1); a1 <- rnorm(1)
v21 <- rnorm(1); v22 <- rnorm(1); v23 <- rnorm(1); a2 <- rnorm(1)

w11 <- rnorm(1); w12 <- rnorm(1); b1 <- rnorm(1)
w21 <- rnorm(1); w22 <- rnorm(1); b2 <- rnorm(1)
w31 <- rnorm(1); w32 <- rnorm(1); b3 <- rnorm(1)

delta <- 0.01
epochs <- 1e4

pb <- txtProgressBar(min = 1, max = epochs, style = 3)
for (t in 1:epochs) {
    for (i in 1:nrow(X)) {
        s1 <- v11*X[i, 1] + v12*X[i, 2] + v13*X[i, 3] + a1
        s2 <- v21*X[i, 1] + v22*X[i, 2] + v23*X[i, 3] + a2
        z1 <- sig(s1); z2 <- sig(s2)

        t1 <- w11*z1 + w12*z2 + b1
        t2 <- w21*z1 + w22*z2 + b2
        t3 <- w31*z1 + w32*z2 + b3
        y1 <- sig(t1); y2 <- sig(t2); y3 <- sig(t3)

        g1 <- (y1 - X[i, 1]) * dsig(t1)
        g2 <- (y2 - X[i, 2]) * dsig(t2)
        g3 <- (y3 - X[i, 3]) * dsig(t3)

        dw11 <- g1 * z1; dw12 <- g1 * z2; db1 <- g1
        dw21 <- g2 * z1; dw22 <- g2 * z2; db2 <- g2
        dw31 <- g3 * z1; dw32 <- g3 * z2; db3 <- g3

        r1 <- g1*w11 + g2*w21 + g3*w31
        r2 <- g1*w12 + g2*w22 + g3*w32
        h1 <- r1 * dsig(s1)
        h2 <- r2 * dsig(s2)

        dv11 <- h1 * X[i, 1]; dv12 <- h1 * X[i, 2]; dv13 <- h1 * X[i, 3]; da1 <- h1
        dv21 <- h2 * X[i, 1]; dv22 <- h2 * X[i, 2]; dv23 <- h2 * X[i, 3]; da2 <- h2

        w11 <- w11 - delta*dw11; w12 <- w12 - delta*dw12; b1 <- b1 - delta*db1
        w21 <- w21 - delta*dw21; w22 <- w22 - delta*dw22; b2 <- b2 - delta*db2
        w31 <- w31 - delta*dw31; w32 <- w32 - delta*dw32; b3 <- b3 - delta*db3

        v11 <- v11 - delta*dv11; v12 <- v12 - delta*dv12; v13 <- v13 - delta*dv13; a1 <- a1 - delta*da1
        v21 <- v21 - delta*dv21; v22 <- v22 - delta*dv22; v23 <- v23 - delta*dv23; a2 <- a2 - delta*da2
  }
  setTxtProgressBar(pb, t)
}
close(pb)

# representação latente do dado de teste

X_tst <- scale(generate_3d_data(100), center = TRUE, scale = FALSE)

latent <- matrix(0, nrow = nrow(X_tst), ncol = 2,
                 dimnames = list(NULL, c("z1", "z2")))

for (i in 1:nrow(X_tst)) {
    s1 <- v11*X_tst[i, 1] + v12*X_tst[i, 2] + v13*X_tst[i, 3] + a1
    s2 <- v21*X_tst[i, 1] + v22*X_tst[i, 2] + v23*X_tst[i, 3] + a2
    latent[i, ] <- c(sig(s1), sig(s2))
}

plot_ly(x = X_tst[,1], y = X_tst[,2], z = X_tst[,3],
        type = "scatter3d", mode = "markers",
        marker = list(size = 2, color = "blue")) |>
    layout(scene = list(xaxis = list(title = "x1"),
                        yaxis = list(title = "x2"),
                        zaxis = list(title = "x3")),
           title = "Dados de teste (3D)")

ggplot(latent, aes(z1, z2)) +
    geom_point(color = "red", size = 1) +
    labs(x = expression(z[1]), y = expression(z[2]),
         title = "Representação latente (2D)") +
    theme_bw()

# erro de reconstrução do dado de teste

error <- numeric(nrow(X_tst))
for (i in 1:nrow(X_tst)) {
    s1 <- v11*X_tst[i, 1] + v12*X_tst[i, 2] + v13*X_tst[i, 3] + a1
    s2 <- v21*X_tst[i, 1] + v22*X_tst[i, 2] + v23*X_tst[i, 3] + a2
    z1 <- sig(s1); z2 <- sig(s2)
    t1 <- w11*z1 + w12*z2 + b1
    t2 <- w21*z1 + w22*z2 + b2
    t3 <- w31*z1 + w32*z2 + b3
    y1 <- sig(t1); y2 <- sig(t2); y3 <- sig(t3)
    error[i] <- mean((c(y1, y2, y3) - X_tst[i, ])^2)
}
cat("RMSE de reconstrução:", sqrt(mean(error)), "\n")
