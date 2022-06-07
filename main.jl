using Plots
using LinearAlgebra
using QuadGK
using Rotations
using Quaternions
using Statistics
using XLSX

# integrates the values
function integrate(value, h)
    value_list = [0.0]
    for i = 2:length(roll)
        push!(value_list, value_list[i-1] + value[i] * h)
    end
    return value_list
end

# Load data
xf = XLSX.readxlsx("data.xlsx")
sh = xf["Sheet1"]
data = sh["A2:F20002"]

# organize data
roll = data[:, :1]
pitch = data[:, :2]
yaw = data[:, :3]
x = data[:, :4]
y = data[:, :5]
z = data[:, :6]

euler_roll = integrate(roll, 1 / 100)
euler_pitch = integrate(pitch, 1 / 100)
euler_yaw = integrate(yaw, 1 / 100)

g = 9.81
acc_roll = []
acc_pitch = []

# converts the x and y accelerations to roll and pitch
for i in 1:length(x)
    push!(acc_roll, asin(x[i] / g))
    push!(acc_pitch, asin(-y[i] / g * cos(acc_roll[i])))
end

# roll is pitch and pitch is roll IDK WHY
# maybe X and Y were swapped accidentally?

temp1 = acc_roll
acc_roll = acc_pitch
acc_pitch = temp1

# goes from a euler to a quaternion
function euler_to_quaternion(roll, pitch, yaw)
    a = RotXYZ(roll, pitch, yaw)
    b = UnitQuaternion(a)
    [b.w; b.x; b.y; b.z]
end

# goes from a quaternion to a euler
function quaternion_to_euler(q)
    b = UnitQuaternion(q[1], q[2], q[3], q[4])
    c = RotXYZ(b)
    [c.theta1
        c.theta2
        c.theta3]
end

# setting inital matrices
x_hat = [[1.0; 0.0; 0.0; 0.0]]
# creates identity matrix
H = Matrix(1I, 4, 4)
# creates a 4x4 identity matrix
P = [Matrix(1.0I, 4, 4)]
# creates a 4x4 matrix with scalar 10^-4
Q = Matrix(10e-4I, 4, 4)
# creates a 4x4 matrix with scalar 10
R = Matrix(10I, 4, 4)

K = []
Z = []
output = []

HT = transpose(H)

dt = 1 / 100
for k in 1:length(euler_pitch)
    x = [0 roll[k] pitch[k] yaw[k]]

    A = I + (dt / 2) * [x[1] -x[2] -x[3] -x[4]
        x[2] x[1] x[4] -x[3]
        x[3] -x[4] x[1] x[2]
        x[4] x[3] -x[2] x[1]]

    # predict the state
    # x_hat[k+1] = A*x_hat[k]
    push!(x_hat, A * x_hat[k])

    # predict the error
    # P[k+1] = A*P[k]*transpose(A)+Q
    push!(P, A * P[k] * transpose(A) + Q)

    # Compute the Kalman gain
    # K[k] = P[k+1]*HT*inv(H*P[k+1]*HT+R)
    push!(K, P[k+1] * HT * inv(H * P[k+1] * HT + R))

    push!(Z, euler_to_quaternion(acc_roll[k], acc_pitch[k], 0))

    # Compute the new estimate
    x_hat[k+1] = x_hat[k+1] + K[k] * (Z[k] - H * x_hat[k+1])

    # compute the new error Covariance
    P[k+1] = P[k+1] - K[k] * H * P[k+1]

    push!(output, quaternion_to_euler(x_hat[k]))
end

output_roll = []
output_pitch = []
output_yaw = []

for out in output
    push!(output_roll, out[1])
    push!(output_pitch, out[2])
    push!(output_yaw, out[3])
end