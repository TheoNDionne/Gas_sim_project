# modules
import numpy as np
import timeit as ti


######################### - CONSTANTS - #########################


GLOBAL_DT = 0.005

WALL_X_MIN = 0
WALL_X_MAX = 1
WALL_Y_MIN = 0
WALL_Y_MAX = 1


######################### - PARTICLE CLASS DEFINITION - #########################


class Particle:
    
    def __init__(self, position, velocity, radius):
        self.position = position
        self.velocity = velocity
        self.radius = radius

    def __repr__(self):
        return f"[x={self.position}, v={self.velocity}, r={self.radius}]"

    def increment_time(self, dt):
        self.position = self.position + self.velocity * dt


######################### - TIME CONTROL - #########################


def increment_all_time_loop(array):
    for obj in array:
        obj.increment_time(GLOBAL_DT)

def increment_all_time(particle):
        particle.increment_time(GLOBAL_DT)

increment_all_time_vectorized = np.vectorize(increment_all_time)


######################### - PARTICLE GENERATION - #########################


def particle_generator(amount, position_range, velocity_range, radius):
    position_delta = position_range[1] - position_range[0]
    velocity_delta = velocity_range[1] - velocity_range[0]
    
    return np.array(
        [
            Particle(
                position_delta * np.random.rand(2) + position_range[0],
                velocity_delta * np.random.rand(2) + velocity_range[0],
                radius
            ) for i in range(amount)
        ]
    )

def generate_particle(position_range, velocity_range, radius):
    position_delta = position_range[1] - position_range[0]
    velocity_delta = velocity_range[1] - velocity_range[0]
    
    return Particle(
                position_delta * np.random.rand(2) + position_range[0],
                velocity_delta * np.random.rand(2) + velocity_range[0],
                radius
            ) 


######################### - INTER-PARTICLE COLLISIONS - #########################


def have_collided(position_1, position_2, radius):
    """
    ACHTUNG: only works for same radius particles
    """
    x_distance = position_2[0] - position_1[0]
    y_distance = position_2[1] - position_1[1]

    distance = np.sqrt( x_distance * x_distance + y_distance * y_distance )

    return distance <= 2*radius

def new_velocity(position, position_prime, velocity, velocity_prime):
    """
    2D collision with dot product.
    https://en.wikipedia.org/wiki/Elastic_collision
    """
    delta_position = position - position_prime
    delta_velocity = velocity - velocity_prime

    norm2 = np.linalg.norm(delta_position) * np.linalg.norm(delta_position)

    return velocity - (np.dot(delta_velocity, delta_position) * delta_position / norm2)

def collide(particle_1, particle_2):
    """
    Checks if two particles have collided with 
    have_collided() and then redistributes their
    speeds using new_velocity(). Warning: minor
    time rollback to keep particles from clipping.
    """
    if have_collided(particle_1.position, particle_2.position, particle_1.radius):

        position_1 = particle_1.position
        position_2 = particle_2.position
        velocity_1 = particle_1.velocity
        velocity_2 = particle_2.velocity

        new_velocity_1 = new_velocity(position_1, position_2, velocity_1, velocity_2)
        new_velocity_2 = new_velocity(position_2, position_1, velocity_2, velocity_1)

        # ANTI-CLIPPING
        particle_1.increment_time(-GLOBAL_DT)
        particle_2.increment_time(-GLOBAL_DT)

        particle_1.velocity = new_velocity_1
        particle_2.velocity = new_velocity_2
        

######################### - COLLISIONS WITH WALLS - #########################

"""
NOT REALLY GOOD YET...
"""
def have_collided_wall(position, radius):
    x_coord = position[0]
    y_coord = position[1]

    return x_coord - radius <= WALL_X_MIN or x_coord + radius >= WALL_X_MAX or y_coord - radius <= WALL_Y_MIN or y_coord + radius >= WALL_Y_MAX

def collide_wall(particle):

    position = particle.position
    x_coord = position[0]
    y_coord = position[1]
    radius = particle.radius

    if x_coord - radius <= WALL_X_MIN or x_coord + radius >= WALL_X_MAX:
        particle.increment_time(-GLOBAL_DT)
        particle.velocity[0] *= -1

    if y_coord - radius <= WALL_Y_MIN or y_coord + radius >= WALL_Y_MAX:
        particle.increment_time(-GLOBAL_DT)
        particle.velocity[1] *= -1


######################### - WRAPPING FUNCTIONS - #########################


def collide_particles_in_array(array):
    array_length = array.size

    for i in range(0, array_length):
        for j in range(i+1, array_length):
            collide(array[i], array[j])

def collide_particles_with_walls(array):
    array_length = array.size

    for i in range(0, array_length):
        collide_wall(array[i])


######################### - SPECIAL - #########################


def collisions_exist(array):
    collision_status = False
    break_condition = False

    array_length = array.size

    # only works if same radius
    radius = array[0].radius

    for i in range(0, array_length):
        for j in range(i+1, array_length):
            if have_collided(array[i].position, array[j].position, radius):
                collision_status = True
                break_condition = True
                break

        if break_condition:
            break
    
    return collision_status


######################### - SIMULATION - #########################


def increment_simulation(array):

    increment_all_time_loop(array)
    collide_particles_in_array(array)
    collide_particles_with_walls(array)

def safe_particle_generator(amount, position_range, velocity_range, radius):
    """
    Keeps particles from spawning clipped.
    User must make sure that particles DO 
    NOT COLLIDE WITH WALLS on their own.
    """
    array = particle_generator(amount, position_range, velocity_range, radius)
    array_length = array.size

    while collisions_exist(array):
        for i in range(0, array_length):
            for j in range(i+1, array_length):
                if have_collided(array[i].position, array[j].position, radius):
                    array[j] = generate_particle(position_range, velocity_range, radius)

    return array

def produce_snapshot(array):
    array_length = array.size
    output_array = np.empty((array_length, 4))

    for i in range(array_length):
        position = array[i].position
        velocity = array[i].velocity
        
        output_array[i, 0] = position[0] # x    
        output_array[i, 1] = position[1] # y
        output_array[i, 2] = velocity[0] # v_x
        output_array[i, 3] = velocity[1] # v_y

    return output_array

def simulate(
    amount, 
    position_range, 
    velocity_range, 
    radius,
    iterations, 
    snapshot_every_n_iterations, 
    notification_every_n_iterations,
    wall_positions=None,
    dt=None
    ):
    """
    wall_positions = [x_min, x_max, y_min, y_max]
    """

    if dt != None:
        global GLOBAL_DT
        GLOBAL_DT = dt

    if wall_positions != None:
        global WALL_X_MIN, WALL_X_MAX, WALL_Y_MIN, WALL_Y_MAX
        WALL_X_MIN = wall_positions[0]
        WALL_X_MAX = wall_positions[1]
        WALL_Y_MIN = wall_positions[2]
        WALL_Y_MAX = wall_positions[3]

    particles = safe_particle_generator(amount, position_range, velocity_range, radius)
    output = np.empty((amount, 4, ((iterations-1) // snapshot_every_n_iterations) + 1))

    for i in range(iterations):
        increment_simulation(particles)

        if i % snapshot_every_n_iterations == 0:
            output[:,:,i//snapshot_every_n_iterations] = produce_snapshot(particles)

        if i % notification_every_n_iterations == 0:
            print(f"Currently at {i} iterations")
    
    print("DONE!")
    return output


######################### - RUN & SAVE SIMULATION - #########################


t1 = ti.default_timer()
result = simulate(100, [0.05, 0.95], [-1,1], 0.01, 100, 2, 50)
t2 = ti.default_timer()
print(t2 - t1)

np.save("Python/Gas_project/gas_simulation_result.npy", result)


######################### - TO-DO - #########################

"""
COMMENTS FOR MYSELF:
 - There seems to be clipping, dafuq is this
"""
