import numpy as np
import Observations
from typing import Tuple

class MultiplicativeExtendedKalmanFilter:
    def __init__(self, roll0:float, pitch0:float, yaw0:float, w0:np.ndarray, x0:np.ndarray, v0:np.ndarray, a0:np.ndarray, salp0:float, sw0:float, sx0:float, sv0:float, sa0:float, sw:float, sa:float, dt:float, g:np.ndarray):
        """
        Constructs a multiplicitive extended kalman filter for estimating the robot pose

        Parameters:
        roll0 (float): initial roll [rad]
        pitch0 (float): initial pitch [rad]
        yaw0 (float): initial yaw, radians
        w0 (3 element numpy array): initial angular velocity [rad/s]
        x0 (3 element numpy array): initial position [m]
        v0 (3 element numpy array): initial velocity [m/s]
        a0 (3 element numpy array): initial acceleration [m/s^2]
        salph0 (float): standard deviation in initial orientation [rad]
        sw0 (float): standard deviation in initial angular velocity [rad/s]
        sx0 (float): standard deviation in initial position [m]
        sv0 (float): standard deviation in initial velocity [m/s]
        sa0 (float): standard deviation in initial acceleration [m/s^2]
        sw (float): standard deviation of the equation w_next = w + 0 * dt
        sa (float): standard deviation of the equation a_next = a + 0 * dt
        dt (float): sample rate
        g (3 element numpy array): gravitational acceleration vector [m/s^2]
        """

        # Initial Rotation
        q0 = self.__quaternion_from_euler_angles(roll0, pitch0, yaw0)

        # World -> Local to Local -> World
        q0[1:] = -q0[1:] # Quaternion conjugate

        # Initialize State
        self.__q = q0
        self.__w = w0
        self.__x = x0
        self.__v = v0
        self.__a = a0

        # Sample rate
        self.__dt = dt

        # Gravity
        self.__g = g

        # Initialize State Covariance
        salp02 = salp0*salp0
        sw02 = sw0*sw0
        sx02 = sx0*sx0
        sv02 = sv0*sv0
        sa02 = sa0*sa0
        self.__P = np.diag(np.array([salp02, salp02, salp02, sw02, sw02, sw02, sx02, sx02, sx02, sv02, sv02, sv02, sa02, sa02, sa02], dtype=np.float64))

        # Process Noise Variables
        self.__sw = sw
        self.__sa = sa

        # Create cached rotation matrix for quick access
        self.__R = self.__quaternion_to_rotation_matrix(self.__q)
    
    def update(self, observations:Observations) -> None:
        """
        Updates the kalman filter with the new observations.
        Should be called every update step.

        Parameters:
        observations (Observations): All observations seen

        Returns:
        None
        """
        # ---A Priori Update Step---

        # State Update
        self.__q += 0.5 * self.__quaternion_multiply(self.__q, self.__w) * self.__dt
        self.__q /= np.linalg.norm(self.__q)

        self.__x += self.__v * self.__dt
        self.__v += self.__a * self.__dt

        # Covariance Update
        F = self.__processTransition()
        Q = self.__processNoiseCovariance()
        self.__P = F @ self.__P @ F.T + Q

        self.__P = self.__ensure_is_positive_semi_definite(self.__P) + (np.eye(15) * 1e-9)

        # ---A Posteriori Update Step---

        # Construct Observation Matrices
        z, H, R = self.__observationMatrices(observations)

        # Innovation covariance
        S = H @ self.__P @ H.T + R
        S = self.__ensure_is_positive_semi_definite(S)

        # Kalman gain
        K = np.linalg.solve(S, H @ self.__P).T

        # Linearized state update
        x = K @ z

        # Covariance update
        self.__P = (np.eye(15, dtype=np.float64) - K @ H) @ self.__P + (np.eye(15) * 1e-9)

        # ---Update nonlinear states---

        # state deviations
        alpha = x[0:3]
        dq = np.append(1, alpha/2)
        dw = x[3:6]
        dx = x[6:9]
        dv = x[9:12]
        da = x[12:15]

        # Rotate q by dq
        self.__q = self.__quaternion_multiply(self.__q, dq)
        self.__q /= np.linalg.norm(self.__q)

        # Increase linear states by their deviations
        self.__w += dw
        self.__x += dx
        self.__v += dv
        self.__a += da

        # Cache rotation matrix representation
        self.__R = self.__quaternion_to_rotation_matrix(self.__q)

    def position(self) -> np.ndarray:
        """
        The current position estimate [m] in world coordinates
        
        Returns:
        3 element numpy array: current position [x, y, z]
        """
        return self.__x
    
    def position_variance(self) -> float:
        """
        Computes the current variance in the position estimate [m]

        Returns:
        float: variance in current position estimate
        """
        return np.linalg.norm(self.__P[6:9, 6:9])
    
    def rotation_matrix(self) -> np.ndarray:
        """
        The current orientation estimate as a rotation matrix
        The rotation matrix is from local to world coordinates
        v_world = R @ v_local

        Returns:
        3x3 numpy matrix : current rotation matrix
        """
        return self.__R
    
    def orientation_variance(self) -> float:
        """
        The variance in the current orientation estimate [rad]

        Returns:
        float: the variance in the current orientation
        """
        return np.linalg.norm(self.__P[0:3, 0:3])

    def velocity(self) -> np.ndarray:
        """
        The current velocity estimate [m/s] in world coordinates
        
        Returns:
        3 element numpy array: current velocity [x, y, z]
        """
        return self.__v
    
    def velocity_variance(self) -> float:
        """
        Computes the current variance in the velocity estimate [m/s]

        Returns:
        float: variance in current velocity estimate
        """
        return np.linalg.norm(self.__P[9:12,9:12])

    def acceleration(self) -> np.ndarray:
        """
        The current acceleration estimate [m/s^2] in world coordinates
        
        Returns:
        3 element numpy array: current acceleration [x, y, z]
        """
        return self.__a
    
    def acceleration_variance(self) -> float:
        """
        Computes the current variance in the acceleration estimate [m/s^2]

        Returns:
        float: variance in current acceleration estimate
        """
        return np.linalg.norm(self.__P[12:15,12:15])
    
    def angular_velocity(self) -> np.ndarray:
        """
        The current angular velocity [rad/s] in local coordinates
        
        Returns:
        3 element numpy array: current angular velocity [x, y, z]
        """
        return self.__w
    
    def angular_velocity_variance(self) -> float:
        """
        Computes the current variance in the angular velocity estimate [rad/s]

        Returns:
        float: variance in current angular velocity estimate
        """
        return np.linalg.norm(self.__P[3:6,3:6])

    def __processNoiseCovariance(self) -> np.ndarray:
        """
        Compute process covariance matrix
            / dt
        Q = | exp(F * (tau-t)) * Qc * exp(F^T * (tau-t)) dtau
            / 0

        F is the jacobian of the continuous time state transition model xdot = F x

        Returns:
        Q: Process Covariance Matrix
        """
        # Code generated by MATLAB Code generation
        wx = self.__w[0]
        wy = self.__w[1]
        wz = self.__w[2]

        dt = self.__dt

        sw = self.__sw
        sa = self.__sa
        dt = self.__dt

        w = np.sqrt(wx*wx+wy*wy+wz*wz)

        Q = np.zeros((15,15))
        if w < 1e-3:
            Q[0, 0] = ((dt*dt*dt)*(sw*sw))/3.0
            Q[0, 3] = ((dt*dt)*(sw*sw))/2.0
            Q[1, 1] = ((dt*dt*dt)*(sw*sw))/3.0
            Q[1, 4] = ((dt*dt)*(sw*sw))/2.0
            Q[2, 2] = ((dt*dt*dt)*(sw*sw))/3.0
            Q[2, 5] = ((dt*dt)*(sw*sw))/2.0
            Q[3, 0] = ((dt*dt)*(sw*sw))/2.0
            Q[3, 3] = dt*(sw*sw)
            Q[4, 1] = ((dt*dt)*(sw*sw))/2.0
            Q[4, 4] = dt*(sw*sw)
            Q[5, 2] = ((dt*dt)*(sw*sw))/2.0
            Q[5, 5] = dt*(sw*sw)
            Q[6, 6] = ((dt*dt*dt*dt*dt)*(sa*sa))/2.0E+1
            Q[6, 9] = ((dt*dt*dt*dt)*(sa*sa))/8.0
            Q[6, 12] = ((dt*dt*dt)*(sa*sa))/6.0
            Q[7, 7] = ((dt*dt*dt*dt*dt)*(sa*sa))/2.0E+1
            Q[7, 10] = ((dt*dt*dt*dt)*(sa*sa))/8.0
            Q[7, 13] = ((dt*dt*dt)*(sa*sa))/6.0
            Q[8, 8] = ((dt*dt*dt*dt*dt)*(sa*sa))/2.0E+1
            Q[8, 11] = ((dt*dt*dt*dt)*(sa*sa))/8.0
            Q[8, 14] = ((dt*dt*dt)*(sa*sa))/6.0
            Q[9, 6] = ((dt*dt*dt*dt)*(sa*sa))/8.0
            Q[9, 9] = ((dt*dt*dt)*(sa*sa))/3.0
            Q[9, 12] = ((dt*dt)*(sa*sa))/2.0
            Q[10, 7] = ((dt*dt*dt*dt)*(sa*sa))/8.0
            Q[10, 10] = ((dt*dt*dt)*(sa*sa))/3.0
            Q[10, 13] = ((dt*dt)*(sa*sa))/2.0
            Q[11, 8] = ((dt*dt*dt*dt)*(sa*sa))/8.0
            Q[11, 11] = ((dt*dt*dt)*(sa*sa))/3.0
            Q[11, 14] = ((dt*dt)*(sa*sa))/2.0
            Q[12, 6] = ((dt*dt*dt)*(sa*sa))/6.0
            Q[12, 9] = ((dt*dt)*(sa*sa))/2.0
            Q[12, 12] = dt*(sa*sa)
            Q[13, 7] = ((dt*dt*dt)*(sa*sa))/6.0
            Q[13, 10] = ((dt*dt)*(sa*sa))/2.0
            Q[13, 13] = dt*(sa*sa)
            Q[14, 8] = ((dt*dt*dt)*(sa*sa))/6.0
            Q[14, 11] = ((dt*dt)*(sa*sa))/2.0
            Q[14, 14] = dt*(sa*sa)
            return Q

        t2 = dt*w
        t3 = dt*dt
        t4 = dt*dt*dt
        t6 = dt*dt*dt*dt*dt
        t7 = sa*sa
        t8 = sw*sw
        t9 = w*w
        t10 = w*w*w
        t12 = w*w*w*w*w
        t14 = w*w*w*w*w*w*w
        t15 = wx*wx
        t16 = wx*wx*wx
        t18 = wx*wx*wx*wx*wx
        t19 = wx*wx*wx*wx*wx*wx*wx
        t20 = wy*wy
        t21 = wy*wy*wy
        t23 = wy*wy*wy*wy*wy
        t25 = wz*wz
        t26 = wz*wz*wz
        t28 = wz*wz*wz*wz*wz
        t5 = t3*t3
        t11 = t9*t9
        t13 = t9*t9*t9
        t17 = t15*t15
        t22 = t20*t20
        t24 = t20*t20*t20
        t27 = t25*t25
        t29 = t25*t25*t25
        t30 = t2*6.0
        t31 = np.cos(t2)
        t32 = np.sin(t2)
        t34 = 1.0/t12
        t36 = t10*t10*t10
        t39 = t15*2.0
        t41 = t20*2.0
        t42 = t25*2.0
        t45 = dt*t7
        t46 = dt*t8
        t50 = t23*wx*2.0
        t51 = t18*wy*2.0
        t52 = t28*wx*2.0
        t53 = t18*wz*2.0
        t54 = t2*t12*wy*2.0
        t55 = t2*t12*wz*2.0
        t56 = t8*wx*wy*2.0
        t57 = t8*wx*wz*2.0
        t58 = t8*wy*wz*2.0
        t65 = t16*t21*4.0
        t66 = t15*t25*4.0
        t67 = t16*t26*4.0
        t75 = t2*t3*t15
        t76 = t2*t3*t20
        t77 = t2*t3*t25
        t79 = 1.0/(t14*2.0)
        t87 = t20*t26*wx*4.0
        t88 = t21*t25*wx*4.0
        t89 = t16*t25*wy*4.0
        t90 = t16*t20*wz*4.0
        t107 = (t2*t2)*t9*t20
        t108 = (t2*t2)*t25
        t109 = t3*t15*t20
        t110 = t3*t15*t25
        t122 = (t3*t7)/2.0
        t123 = (t4*t7)/3.0
        t124 = (t4*t7)/6.0
        t125 = t3*t8*t21*wx
        t126 = t3*t8*t16*wy
        t127 = t3*t8*t26*wx
        t128 = t3*t8*t16*wz
        t129 = t3*t8*t26*wy
        t130 = t3*t8*t21*wz
        t144 = (t6*t7)/2.0E+1
        t158 = t3*t8*t25*wx*wy
        t159 = t3*t8*t20*wx*wz
        t160 = t3*t8*t15*wy*wz
        t176 = t4*t8*t15*t20
        t177 = t4*t8*t15*t25
        t178 = t4*t8*t20*t25
        t33 = 1.0/t11
        t35 = 1.0/t13
        t40 = t17*2.0
        t43 = t27*2.0
        t44 = -t30
        t48 = t32*6.0
        t60 = t22*wx*wz*2.0
        t61 = -t50
        t62 = -t51
        t63 = t3*t17
        t64 = t20*t39
        t68 = t25*t41
        t69 = -t54
        t70 = -t55
        t71 = -t56
        t73 = -t58
        t74 = t27*wx*wy*-2.0
        t78 = t34/3.0
        t80 = -t65
        t81 = t16*t46*2.0
        t82 = t15*t46*6.0
        t83 = t21*t46*2.0
        t84 = t20*t46*6.0
        t85 = t26*t46*2.0
        t86 = t25*t46*6.0
        t91 = 1.0/(t36*2.0)
        t97 = (t2*t2)*t11*wx*wy
        t98 = (t2*t2)*t11*wx*wz
        t99 = t31*t50
        t100 = t31*t51
        t105 = -t88
        t106 = -t89
        t111 = t15*t31*-2.0
        t112 = t17*t31*-2.0
        t113 = t20*t31*-2.0
        t114 = t25*t31*-2.0
        t115 = t27*t31*-2.0
        t116 = t41*t46*wx
        t117 = t39*t46*wy
        t118 = t42*t46*wx
        t119 = t39*t46*wz
        t120 = t42*t46*wy
        t121 = t41*t46*wz
        t131 = t28*t31*wx*-2.0
        t132 = t18*t31*wz*-2.0
        t133 = t31*t56
        t135 = t31*t58
        t143 = (t5*t7)/8.0
        t146 = t31*t65
        t151 = t8*t19*t32*2.0
        t152 = t8*t23*t32*2.0
        t154 = t8*t28*t32*2.0
        t156 = t10*t21*t32*2.0
        t157 = t10*t26*t32*2.0
        t161 = t22*t31*wx*wz*-2.0
        t163 = t31*t88
        t164 = t31*t89
        t167 = t8*t24*t32*wx*2.0
        t169 = t8*t29*t32*wx*2.0
        t171 = t8*t22*t32*wz*2.0
        t172 = t10*t32*t39*wy
        t173 = t10*t32*t39*wz
        t174 = t10*t32*t42*wy
        t175 = t10*t32*t41*wz
        t180 = t15*t25*t31*-4.0
        t181 = t16*t26*t31*-4.0
        t188 = t20*t26*t31*wx*-4.0
        t189 = t16*t20*t31*wz*-4.0
        t194 = t8*t15*t21*t32*4.0
        t197 = t8*t15*t26*t32*4.0
        t200 = t8*t20*t26*t32*4.0
        t201 = t8*t21*t25*t32*4.0
        t202 = t8*t32*t66*wy
        t203 = t8*t15*t20*t32*wz*4.0
        t215 = t8*t16*t20*t25*t32*1.2E+1
        t47 = t33*t33
        t59 = t43*wx*wy
        t142 = -t98
        t150 = t8*t15*t48
        t153 = t8*t20*t48
        t155 = t8*t25*t48
        t166 = t8*t32*t40*wy
        t168 = t8*t32*t40*wz
        t170 = t8*t32*t43*wy
        t179 = t20*t111
        t182 = t25*t113
        t195 = t8*t16*t22*t48
        t198 = t8*t16*t27*t48
        t217 = t44+t48+t75+t76+t77
        t218 = t39+t41+t108+t111+t113
        t223 = t41+t42+t63+t109+t110+t113+t114
        t225 = t71+t85+t119+t121+t125+t126+t133+t158
        t226 = t73+t81+t116+t118+t129+t130+t135+t160
        t232 = -t10*(t57+t83+t117+t120-t127-t128-t159-t8*t31*wx*wz*2.0)
        t233 = -t12*(t58+t81+t116+t118-t129-t130-t160-t8*t31*wy*wz*2.0)
        t136 = t31*t59
        t196 = t18*t153
        t199 = t18*t155
        t204 = t27*t153*wx
        t205 = t22*t155*wx
        t219 = t8*t78*t217*wx*wy
        t220 = t8*t78*t217*wx*wz
        t221 = t8*t78*t217*wy*wz
        t222 = (t8*t33*t218)/2.0
        t224 = (t8*t33*t223)/2.0
        t227 = t152+t166+t170+t194+t201+t202
        t230 = t10*t225
        t231 = t12*t226
        t235 = t40+t43+t64+t66+t68+t107+t112+t115+t179+t180+t182
        t241 = t52+t53+t60+t67+t69+t87+t90+t131+t132+t142+t156+t161+t172+t174+t181+t188+t189
        t236 = (t8*t35*t235)/2.0
        t237 = t151+t167+t169+t195+t196+t198+t199+t204+t205+t215
        t239 = t61+t62+t70+t74+t80+t97+t99+t100+t105+t106+t136+t146+t157+t163+t164+t173+t175
        t240 = t227+t232
        t245 = ((t154+t168+t171+t197+t200+t203-t230)*(-1.0/2.0))/t14
        t246 = (t8*t47*t241)/2.0
        t243 = t79*t240
        t244 = (t8*t47*t239)/2.0
        t247 = -t246
        t248 = t233+t237
        t251 = (t231-t237)/(t36*2.0)
        t250 = t91*t248
        Q[0, 0] = t34*(t153+t155-w*(t84+t86+t176+t177+t4*t8*t17))*(-1.0/3.0)
        Q[0, 1] = t219
        Q[0, 2] = t220
        Q[0, 3] = t224
        Q[0, 4] = t245
        Q[0, 5] = t243
        Q[1, 0] = t219
        Q[1, 1] = t34*(t150+t155-w*(t82+t86+t176+t178+t4*t8*t22))*(-1.0/3.0)
        Q[1, 2] = t221
        Q[1, 3] = t244
        Q[1, 4] = t236
        Q[1, 5] = t251
        Q[2, 0] = t220
        Q[2, 1] = t221
        Q[2, 2] = t34*(t150+t153-w*(t82+t84+t177+t178+t4*t8*t27))*(-1.0/3.0)
        Q[2, 3] = t247
        Q[2, 4] = t250
        Q[2, 5] = t222
        Q[3, 0] = t224
        Q[3, 1] = t244
        Q[3, 2] = t247
        Q[3, 3] = t46
        Q[4, 0] = t245
        Q[4, 1] = t236
        Q[4, 2] = t250
        Q[4, 4] = t46
        Q[5, 0] = t243
        Q[5, 1] = t251
        Q[5, 2] = t222
        Q[5, 5] = t46
        Q[6, 6] = t144
        Q[6, 9] = t143
        Q[6, 12] = t124
        Q[7, 7] = t144
        Q[7, 10] = t143
        Q[7, 13] = t124
        Q[8, 8] = t144
        Q[8, 11] = t143
        Q[8, 14] = t124
        Q[9, 6] = t143
        Q[9, 9] = t123
        Q[9, 12] = t122
        Q[10, 7] = t143
        Q[10, 10] = t123
        Q[10, 13] = t122
        Q[11, 8] = t143
        Q[11, 11] = t123
        Q[11, 14] = t122
        Q[12, 6] = t124
        Q[12, 9] = t122
        Q[12, 12] = t45
        Q[13, 7] = t124
        Q[13, 10] = t122
        Q[13, 13] = t45
        Q[14, 8] = t124
        Q[14, 11] = t122
        Q[14, 14] = t45

        return Q
    
    def __processTransition(self) -> np.ndarray:
        """
        Computes the process transition matrix: the jacobian of the discrete time state transition

        Returns:
        F: process transition matrix
        """
        # Code generated by MATLAB code generation
        F = np.zeros((15,15), dtype=np.float64)
        
        F[0, 1] = self.__w[2]
        F[0, 2] = -self.__w[1]
        F[0, 3] = 1.0
        F[1, 0] = -self.__w[2]
        F[1, 2] = self.__w[0]
        F[1, 4] = 1.0
        F[2, 0] = self.__w[1]
        F[2, 1] = -self.__w[0]
        F[2, 5] = 1.0
        F[6, 9] = 1.0
        F[7, 10] = 1.0
        F[8, 11] = 1.0
        F[9, 12] = 1.0
        F[10, 13] = 1.0
        F[11, 14] = 1.0

        F = np.eye(15, dtype=np.float64) + F * self.__dt
        return F
        
    def __observationMatrices(self, observations:Observations.Observations) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates matrices based on the observations for the a posteriori kalman update

        Parameters:
        observations (Observations): observations

        Returns:
        z: observation residuals
        H: observation matrix
        R: observation covariance matrix
        """
        z = np.ndarray((0),dtype=np.float64)
        H = np.ndarray((0,15),dtype=np.float64)
        r = np.ndarray((0),dtype=np.float64)


        # Current Rotation Matrix
        # From world to body = q^-1
        R = self.__quaternion_to_rotation_matrix(self.__q)
        R = R.T

        # Gyroscope
        z_gyro = observations.imu.gyro - self.__w
        H_gyro = np.zeros((3,15), dtype=np.float64)
        H_gyro[:, 3:6] = np.eye(3)

        z = np.concatenate((z, z_gyro))
        H = np.concatenate((H, H_gyro), axis=0)
        r = np.concatenate((r, np.ones(3) * observations.imu.gyro_var))

        # Accelerometer
        z_accel = observations.imu.accel - self.__a - self.__g
        H_accel = np.zeros((3,15), dtype=np.float64)
        H_accel[:,0:3] = self.__vector_to_skew_symmetric(R @ (self.__a + self.__g))
        H_accel[:,12:15] = R

        z = np.concatenate((z, z_accel))
        H = np.concatenate((H, H_accel), axis=0)
        r = np.concatenate((r, np.ones(3) * observations.imu.accel_var))

        # Linear odometry
        z_lin_odo = np.array([observations.odo.lin_speed,0,0]) - R @ self.__v
        H_lin_odo = np.zeros((3,15), dtype=np.float64)
        H_lin_odo[:,0:3] = -self.__vector_to_skew_symmetric(R @ self.__v)
        H_lin_odo[:,9:12] = R

        z = np.concatenate((z, z_lin_odo))
        H = np.concatenate((H, H_lin_odo), axis=0)
        s1 = observations.odo.lin_speed_var
        s2 = observations.odo.lin_speed_off_axis_var
        r = np.concatenate((r, np.array([s1,s2,s2], dtype=np.float64)))

        # Angular odometry
        z_ang_odo = np.array([0,0,observations.odo.ang_speed])
        H_ang_odo = np.zeros((3,15), dtype=np.float64)
        H_ang_odo[:,3:6] = np.eye(3, dtype=np.float64)

        z = np.concatenate((z, z_ang_odo))
        H = np.concatenate((H, H_ang_odo), axis=0)
        s1 = observations.odo.ang_speed_var
        s2 = observations.odo.ang_speed_off_axis_var
        r = np.concatenate((r, np.array([s2,s2,s1], dtype=np.float64)))


        # Point Observations
        for point in observations.points:
            Vxy = R @ (point.world_coord - self.__x)

            z_point = point.local_coord - Vxy
            H_point = np.zeros((3,15), dtype=np.float64)
            H_point[:,0:3] = self.__vector_to_skew_symmetric(Vxy)
            H_point[:,6:9] = -R

            z = np.concatenate((z, z_point))
            H = np.concatenate((H, H_point), axis=0)
            r = np.concatenate((r, np.ones(3) * point.var))
        
        # Direction Observations
        for dir in observations.directions:
            Ve = R @ dir.world_dir

            z_dir = dir.local_dir - Ve
            H_dir = np.zeros((3,15), dtype=np.float64)
            H_dir[:,0:3] = self.__vector_to_skew_symmetric(Ve)

            z = np.concatenate((z, z_dir))
            H = np.concatenate((H, H_dir), axis=0)
            r = np.concatenate((r, np.ones(3) * dir.var))

        return z, H, np.diag(r)
 
    def __vector_to_skew_symmetric(self, v) -> np.ndarray:
        """
        Create a skew skymmetric matrix from a vector

        Parameters:
        v (3 element numpy array): vector

        Returns:
        3x3 numpy matrix: skew symmetric matrix
        """
        x, y, z = v
        M = np.zeros((3,3), dtype=np.float64)
        M[0,1] = -z
        M[1,0] =  z
        M[0,2] =  y
        M[2,0] = -y
        M[1,2] = -x
        M[2,1] =  x

        return M

    def __ensure_is_positive_semi_definite(self, M:np.ndarray) -> np.ndarray:
        """
        Ensure the matrix M is positive semi definite: M is symmetric and does not have negative entries

        Parameters:
        M (numpy array): matrix to make positive semi definite

        Returns:
        M (numpy array): corrected positive semi definite matrix
        """
        M  = (M + M.T) / 2
        L = np.linalg.cholesky(M)
        return L @ L.T

    def __quaternion_multiply(self, q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
        """
        Multiply quaternions. If the inputs have 4 elements, they are treated as quaternions. If the inputs have 3 elements, they are treated as vectors and a zero value is used for the real component

        Parameters:
        q1 (numpy array): first quaternion to multiply
        q2 (numpy array): second quaternion to multiply

        Returns:
        q (numpy array): quaternion multiplication of q1 and q2
        """
        if np.size(q1) == 4:
            w0, x0, y0, z0 = q1
        else:
            w0 = 0
            x0, y0, z0 = q1
        
        if np.size(q2) == 4:
            w1, x1, y1, z1 = q2
        else:
            w1 = 0
            x1, y1, z1 = q2
        
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                          x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                          x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def __quaternion_to_rotation_matrix(self, q:np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix

        Parameters:
        q (numpy array): quaternion to convert

        Returns:
        R (numpy array): rotation matrix
        """
        r, i, j, k = q

        s = 2/(r*r + i*i + j*j + k*k)

        return np.array([[1 - s*(j*j+k*k),     s*(i*j-k*r),     s*(i*k+j*r)],
                         [    s*(i*j+k*r), 1 - s*(i*i+k*k),     s*(j*k-i*r)],
                         [    s*(i*k-j*r),     s*(j*k+i*r), 1 - s*(i*i+j*j)]])
        
    def __quaternion_from_euler_angles(self, roll:float, pitch:float, yaw:float) -> np.ndarray:
        """
        Convert euler angles to quaternion. 
        Roll: Rotation around x axis
        Pitch: Rotation around y axis
        Yaw: Rotation around z axis
        Rotation order is RPY: Yaw * Pitch * Roll

        Parameters:
        roll (float): rotation around x axis [rad]
        pitch (float): rotation around y axis [rad]
        yaw(float): rotation around z axis [rad]
        """
        q_roll =  np.array([np.cos(roll/2),  np.sin(roll/2), 0,               0            ])
        q_pitch = np.array([np.cos(pitch/2), 0,              np.sin(pitch/2), 0            ])
        q_yaw =   np.array([np.cos(yaw/2),   0,              0,               np.sin(yaw/2)])

        return self.__quaternion_multiply(q_yaw, self.__quaternion_multiply(q_pitch, q_roll))
    

