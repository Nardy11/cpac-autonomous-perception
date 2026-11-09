import rospy
import math
import pygame
from sensor_msgs.msg import LaserScan

# Initialize the global distance list
distance_list = LaserScan()

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()

def calculate_point_on_circle(radius, angle, center=(400, 400)):
    cx, cy = center
    x = int(cx + radius * math.cos(math.radians(90 - angle)))
    y = int(cy + radius * math.sin(math.radians(90 - angle)))
    return x, y

def plot_lidar_data():
    """
    Function to plot LIDAR data points on a circle.
    """
    global distance_list
    plot_list = []
    center = (400, 400)  # Center of the circle
    
    # Iterate over the distances and calculate the corresponding points
    for i, distance in enumerate(distance_list.ranges):
        if distance < 10000:
            radius = distance   # Scale radius
            angle = i  # Angle in degrees
            x_point, y_point = calculate_point_on_circle(radius, angle, center)
            plot_list.append((x_point, y_point))
    
    # Plot the points
    screen.fill((255, 255, 255))  # Clear the screen
    pygame.draw.circle(screen, (255, 0, 0), center, 5)  # Plot the center
    for point in plot_list:
        pygame.draw.circle(screen, (0, 0, 255), point, 3)  # Plot each LIDAR point
    pygame.display.flip()

def lidarCallback(data):
    """
    Callback function for processing LIDAR data.

    Args:
        data: Received LIDAR data message.
    """
    global distance_list
    distance_list = data
    #print("Received LIDAR data:", distance_list)  # Debug statement
    plot_lidar_data()  # Call the plotting function each time new data is received


# Initialize the ROS node
rospy.init_node('lidar_plotter')

# Subscribe to the LIDAR topic
subLidar = rospy.Subscriber('/scan', LaserScan, lidarCallback)

# Main loop
running = True
while running and not rospy.is_shutdown():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    plot_lidar_data()
    rospy.sleep(0.1)  # Adjust sleep time as needed
    clock.tick(10)  # Limit frame rate to 10 FPS

pygame.quit()

