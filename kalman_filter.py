import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def getMeasurement(updateNumber):
    if updateNumber == 1:
        getMeasurement.currentPosition = 0
        getMeasurement.currentVelocity = 60

        dt = 0.1

        w= 8*np.random.randn(1)
        v=8*np.random.randn(1)

        z=getMeasurement.currentPosition + (getMeasurement.currentVelocity)*dt + v
        getMeasurement.currentPosition = z-v
        getMeasurement.currentVelocity = 60 + w

        return [z,getMeasurement.currentPosition, getMeasurement.currentVelocity]
    
    else:
        return [0,0,0]

def filter(z,updateNumber):
    dt = 0.1
    #initialize the state
    if updateNumber ==1 :
        # x -- State vector/variable -- position = 0, velocity = 20 (initialized) 
        # set based on edcuated guess or info about the system's state 
        filter.x = np.array([[0],[20]]) 
        # P -- Covariance in estimated state variable -- depends on the uncertainty with initial state estimate
        filter.P = np.array([[5,0],[0,5]]) 
        # A -- state transition -- how the state of the system evolves over time
        filter.A = np.array([[1,dt],[0,1]])
        # H -- state to measurement matrix -- taking the estimated state to predict the observed measurement
        filter.H = np.array([[1,0]]) #the value means we are only measuring the position of the system.
        # transpose of H
        filter.HT = np.array([[1],[0]])
        # R -- measurement noise covariance  -- value represents the variance of the noise in measurements.
        filter.R = 10 #higher value means higher degree of uncertainity
        # Q -- process noise covariance 
        filter.Q = np.array([[1,0],
                             [0,3]]) #higher number in diagonal elements means higher degree of uncertainity. 1 - position, 3 - velocity
        
        #predicting the state forward -- how position and velocity are expected to change over time
        x_p = filter.A.dot(filter.x)
        # predict covariance forward
        P_p = filter.A.dot(filter.P).dot(filter.A.T)+filter.Q

        '''S --innovation covariance. 
        It is the total uncertainty associated with the difference between predicted and actual measurement.'''
        S = filter.H.dot(P_p).dot(filter.HT) + filter.R 
        
        '''K -- kalman gain.
        how much weight to give to the prediction vs the measurement when updating the state estimate'''
        K = P_p.dot(filter.HT).dot(np.linalg.inv(S))

        #estimate state
        residual = z - filter.H.dot(x_p) #difference between actual and predicted measurement
        #update step
        filter.x = x_p + K*residual


        #state covariance -- update of state covariance matrix
        filter.P = P_p - K.dot(filter.H).dot(P_p)


    return [filter.x[0],filter.x[1],filter.P]



def test_filter():
    dt = 0.1
    t = np.linspace(1,10,num=300) # 1D array of evenly spaced numbers
    numOfMeasurements = len(t)

    measTime = []
    measPos = []
    measDifPos = []
    estDifPos = []
    estPos =[]
    estVel = []
    posBound3Sigma =[] # a rule in stats

    for k in range(1,numOfMeasurements):
        z= getMeasurement(k)
        #call filter and return new state
        f = filter(z[0],k)
        #saving the state so that it could be plotted
        measTime.append(k)
        measPos.append(z[0])
        measDifPos.append(z[0]-z[1])
        estDifPos.append(f[0]-z[0])
        estPos.append(f[0])
        estVel.append(f[1])
        posVar = f[2]
        posBound3Sigma.append(3*np.sqrt(posVar[0][0]))


    return [measTime,measPos,estPos,estVel,measDifPos,estDifPos,posBound3Sigma]
    


#plotting measurement error and estimate error relative to the actual position

t = test_filter()
plot1 = plt.figure(1)
plt.scatter(t[0],t[1])
plt.plot(t[0],t[2])
plt.ylabel('Position')
plt.xlabel('Time')
plt.grid(True)


plot2=plt.figure(2)
plt.plot(t[0],t[3])
plt.ylabel('Velocity (m/s)')
plt.xlabel('Update Number')
plt.title('Velocity update on each measurement update \n',fontweight = 'bold')
plt.legend(['Estimate'])
plt.grid(True)


plot3=plt.figure(3)
plt.scatter(t[0],t[4],color ='red')
plt.plot(t[0],t[5])
plt.legend(['Estimate','Measurement'])
plt.title('Position Errors on each measurement update \n',fontweight='bold')
#plt.plot(t[0],t[6])
plt.ylabel('Position Error (m)')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0,300])

plt.show()


