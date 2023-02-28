import random
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import time as tm

# constant definition
COUNTRY_LENGTH = 1000
COUNTRY_WIDTH = 1000
HUMAN_RADIUS = 0.01
QUA_RADIUS = 0.05
EFFECTIVE_DIST = 0.1

# debug
ERR_INIT_COUNTRY = "Failed to initialise country"
ERR_INIT_PERSON  = "Failed to initialise person"

# helper functions
def dist(position1, position2):
	dist = np.sqrt((position1[0]-position2[0])**2 + (position1[1]-position2[1])**2)
	return dist

def x_y_outside(pos, wall):
	'''
	input: pos:2d np.vector
	wall: 4 tuple contains for the corner of the wall
	in the order of xmin, xmax, ymin, ymax
	'''
	(xmin,xmax,ymin,ymax) = wall
	x_outside = False
	y_outside = False
	if pos[0] < xmin or pos[0] > xmax:
		x_outside = True
	if pos[1] < ymin or pos[1] > ymax:
		y_outside = True

	return (x_outside,y_outside)

def locate(pos, area1, area2, area3, area4):
	if not x_y_outside(pos,area1)[0] and not x_y_outside(pos,area1)[1]:
		return area1
	if not x_y_outside(pos,area2)[0] and not x_y_outside(pos,area2)[1]:
		return area2
	if not x_y_outside(pos,area3)[0] and not x_y_outside(pos,area3)[1]:
		return area3
	if not x_y_outside(pos,area4)[0] and not x_y_outside(pos,area4)[1]:
		return area4

class country:

	'''
	input: name: string, restrictions: dictionary of keys
	being different restrictions and values being booleans, population: int
	enforement_rate: float between 0 and 1
	location: 4-tuple indicating its position on the map consists of xMin, xMax, yMin, yMax
	'''

	def __init__(self, name, restrictions, population, enforement_rate, location=(-0.8,0.8,-0.8,0.8)):

		self.name = name
		self.restrictions = restrictions
		self.enforement_rate = enforement_rate
		self.population = population

		# location
		self.location = location
		centre = [(self.location[0]+self.location[1])/2, \
						(self.location[2]+self.location[3])/2]
		self.area1 = [self.location[0],centre[0],self.location[2],centre[1]]
		self.area2 = [centre[0],self.location[1],self.location[2],centre[1]]
		self.area3 = [self.location[0],centre[0],centre[1],self.location[3]]
		self.area4 = [centre[0],self.location[1],centre[1],self.location[3]]

		self.areas = [self.area1,self.area2,self.area3,self.area4]

		# randomly generate a bunch of people with certain size
		self.people = []
		self.create_people()

		# cases
		self.total = 0
		self.totals = []
		self.death = 0
		self.deaths = []
		self.daily = 0
		self.dailies = []

		for person in self.people:
			if person.is_sick:
				self.total += 1

		# time
		self.day_count = 0

		# visualisation
		self.visualisation = AreaVisualisation(location)

	def create_people(self):
		for i in range(self.population):
			if random.random() > self.enforement_rate:
				is_legal = False
			else:
				is_legal = True
			position = np.array([random.uniform\
			(self.location[0],self.location[1]),\
			 random.uniform(self.location[2],self.location[3])])
			if i == 0:
				self.people.append \
				(person(position, is_legal=is_legal, is_sick=True))
			self.people.append(person(position, is_legal=is_legal))

	def cal_infect_rate(self, person):

		'''if there are sick people within dist<(), it is possible
		to get sick with base rate of (0.2), its square grows propotional to
		the distance of the sick people'''

		base_rate = 0.1
		# might not be factually accurate but doesn't matter in this case

		pb_infect = 0
		location1 = locate(person.position,self.area1,self.area2,self.area3,self.area4)
		for person2 in self.people:
			if not person2.is_sick or not person2.is_alive: # also counts for the case when it's the same as person
				continue
			if locate(person2.position,self.area1,self.area2,self.area3,self.area4) \
				!= location1:
				continue
			dis = dist(person.position,person2.position)
			if not dis < EFFECTIVE_DIST:
				continue

			if 0 == pb_infect:
				pb_infect = base_rate/(100*dis**2)
			else:
				# the rate of getting sick is equal to
				# 1 - the probability of not getting sick
				pb_infect = 1 - (1 - pb_infect)*(1-(base_rate/dis**2))
				if pb_infect > 1:
					return 1

		return pb_infect

	def infect_people(self):
		for person in self.people:
			if person.is_immune or person.is_sick:
				continue

			infect_rate = self.cal_infect_rate(person)
			if random.random() < infect_rate:
				person.is_sick = True
				self.total += 1
				self.daily += 1
				if random.random() > 0.05:
					person.is_symptomatic = True
		return


	def detect_collision(self, person, delta):
		# hits people
		if not person.is_quarantined:
			for person2 in self.people:
				if not person2.is_alive:
					continue
				if person2 != person and \
				dist(person.position, person2.position) <= \
				1.2*(person.radius+person2.radius):
						person.update_velocity(person2)

			# hits wall

			next_pos = person.position + person.velocity*delta

			# area_restrictions
			if self.restrictions['area_restrictions']:
				if random.random() < self.enforement_rate:
					area_wall = locate(person.position,self.area1,self.area2,self.area3,self.area4)
					x_out,y_out = x_y_outside(next_pos, area_wall)
				else:
					x_out,y_out = x_y_outside(next_pos, self.location)
			else:
				x_out,y_out = x_y_outside(next_pos, self.location)

			if x_out:
				person.velocity = np.array([-person.velocity[0],person.velocity[1]])
			if y_out:
				person.velocity = np.array([person.velocity[0],-person.velocity[1]])

		return

	def apply_restrictions(self):

		if self.restrictions['quarantine']:
			for person in self.people:
				if person.is_legal:
					if person.is_quarantined and not person.is_sick:
						person.is_quarantined = False
						person.radius = HUMAN_RADIUS

					if person.is_symptomatic:
						person.is_quarantined = True
						# person.radius = QUA_RADIUS

		if self.restrictions['reduce_travelling']:
			self.restrictions['reduce_travelling'] = False
			for person in self.people:
				if person.is_legal:
					person.velocity *= 0.2

		if self.restrictions['area_restrictions']:
			for area in self.areas:
				AreaVisualisation(area).Draw()
                

	def update_people(self, delta):
		self.apply_restrictions()
		for person in self.people:
			if person.is_alive:
				self.detect_collision(person, delta)
			death_count, recorver_count = person.update(delta)
			self.death += death_count
			self.total -= (death_count+recorver_count)

	def update_data(self):
		count = 1
        
		if not count // 5:
			self.day_count += 1
			print(self.daily)
			self.dailies.append(self.daily)
			self.totals.append(self.total)
			self.deaths.append(self.death)
			self.daily = 0

	def update(self, delta):
		self.infect_people()
		self.update_people(delta)

		for person in self.people:
			person.draw()

		self.update_data()


	def plot_cases_vs_time(self):
		times = np.arange(self.day_count)

		fig, cases = plt.subplots(3)
		fig.suptitle('Cases vs Time(days)')

		cases[0].plot(times, self.totals, label='total vs times')
		cases[0].set_title('total vs times')

		cases[1].plot(times, self.dailies,label='daily vs times')
		cases[1].set_title('daily vs times')

		cases[2].plot(times, self.deaths, label='deaths vs times')
		cases[2].set_title('death vs times')

		plt.legend()
		plt.show()

	def draw(self):

		self.visualisation.Draw()

class person:

	'''type of input: location: string, age: int pb_travel:
	float in [0,1], travel_des: string (name of a country in the country dictionary)
	all is_ variables should be booleans
	'''

	def __init__(self, position,
	is_legal=True, is_sick=False, radius=HUMAN_RADIUS):

		self.position = position

		self.speed = 0.2
		x = random.uniform(-self.speed,self.speed)
		abs_y = np.sqrt(self.speed**2-x**2)
		y = random.choice([-abs_y,abs_y])
		self.velocity = np.array([x,y])

		self.age = random.normalvariate(45, 14)
		self.is_legal = is_legal
		self.radius = radius

		# health statues
		self.is_alive = True # healthy ppl white
		self.is_sick = is_sick
		if is_sick:
			self.is_symptomatic = True
		else:
			self.is_symptomatic = False
		self.days_since_infected = 0
		self.is_quarantined = False # speed = 0

		# yellow for sick but not is_symptomatic n red otherwise
		# self.pb_die_today = 0 # float between 0 and 1
		self.is_immune = False # green

		# visualisation
		self.colour = self.find_colour()
		self.visualisation = PersonVisualisation(self.colour)

	def update_health(self,delta):

		''' people might die or recorver the probability of recovering
		on their age and how long they have been sick for'''

		death_count = 0
		recorver_count = 0
		if self.is_alive and self.is_sick :

			self.days_since_infected += 0.2
			if self.is_symptomatic:
				if self.age < 40:
					if random.random() < 0.0002:
						self.is_alive = False
						death_count +=1
				elif self.age < 50:
					if random.random() < 0.001:
						self.is_alive = False
						death_count +=1
				elif self.age < 60:
					if random.random() < 0.0014:
						self.is_alive = False
						death_count +=1
				elif self.age < 70:
					if random.random() < 0.0018:
						self.is_alive = False
						death_count +=1
				else:
					if random.random() < 0.00296:
						self.is_alive = False
						death_count +=1

			elif self.days_since_infected > 14:
				self.is_symptomatic = True

			if self.days_since_infected >= 21:
				if random.random() < 0.1:
					self.is_sick = False
					self.is_immune = True
					recorver_count += 1
					self.is_symptomatic = False

		return death_count, recorver_count

	def update_velocity(self, another):
		self.velocity[0] = -another.velocity[0]
		self.velocity[1] = -another.velocity[1]

	def update_position(self, delta):
		if self.is_quarantined or not self.is_alive:
			return
		self.position += delta*self.velocity

	def update(self, delta):

		self.update_position(delta)
		death_count,recorver_count = self.update_health(delta)
		# visualisation
		self.colour = self.find_colour()
		self.visualisation = PersonVisualisation(self.colour)

		return death_count, recorver_count
	def find_colour(self):

		colour = [255,255,255]

		if not self.is_alive:
			return [100,20,100]

		if self.is_sick:

			if self.is_quarantined:
				colour = [0,0,255]
			elif self.is_symptomatic:
				colour = [255,0,0]
			else:
				colour = [255,255,0]
		elif self.is_immune:
			colour = [0,255,0]

		return colour

	def draw(self):

		self.visualisation.Draw(self.position, HUMAN_RADIUS)

# the following code has been taken and modified from lab2Visualisation.py

resolution = 800

def Execute(updateFunction, runTime):

	'''
	input: updateFunction: callable function,
	runTime: time in seconds of how long the pygame window should run for
	'''

	global window
	resolution = 800 

	window = pg.display.set_mode((int(resolution*2),resolution))
	window.fill([0,0,0])

	lastFrameTime = tm.process_time()
	totalTime = 0
	while totalTime < runTime:

		timeStep = tm.process_time() - lastFrameTime

		lastFrameTime = tm.process_time()

		window.fill([0,0,0])

		# print(timeStep)
		updateFunction(timeStep)

		pg.display.update()
		totalTime += timeStep

	pg.display.quit()

class PersonVisualisation:

	def __init__(self, colour=[255,255,255]):
		self.colour = colour
		self.pixelPos = [400,400]
		self.pixelRad = 0

	def Draw(self, position, radius):

		self.pixelRad = round(resolution*radius/2.0)

		self.pixelPos = [int(round(resolution*(position[0] + 1.0)/2.0)),
		 int(round(resolution*(position[1] + 1.0)/2.0))]

		pg.draw.circle(window,self.colour,self.pixelPos,self.pixelRad)

class AreaVisualisation:

	def __init__(self, location, colour=[255,255,255]):
		self.colour = colour

		(xMin, xMax, yMin, yMax) = location
		self.topRightX = round(resolution*(xMin + 1.0)/2.0)
		self.topRightY = round(resolution*(yMin + 1.0)/2.0)
		self.width = round(resolution*(xMax-xMin)/2.0)
		self.height = round(resolution*(yMax-yMin)/2.0)

	def Draw(self):

		pg.draw.rect(window,self.colour,[self.topRightX,self.topRightY,self.width, self.height],1)


def update_world(delta):
	'''updates the entire world for each time step delta'''

	for c in countries:
		c.draw()
	for c in countries:
		c.update(delta)


# intialise the world
# default restrictions

no_res = {'quarantine': False, 'reduce_travelling': False, 'area_restrictions': False}
quarantine = {'quarantine': True, 'reduce_travelling': False, 'area_restrictions': False}
red_tra = {'quarantine': False, 'reduce_travelling': True, 'area_restrictions': False}
area_res = {'quarantine': False, 'reduce_travelling': False, 'area_restrictions': True}
all = {'quarantine': True, 'reduce_travelling': True, 'area_restrictions': True}
c1 = country('country1',no_res, 200, 0, [-1,0.99,-1,1])
# c1 = country('country1',red_tra, 200, 0, [-1,0.99,-1,1])
# c2 = country('count2', red_tra, 200, 0.7, [1.01,3,-1,1])
# c2 = country('count2', red_tra, 200, 0.95, [1.01,3,-1,1])
# c2 = country('count2', area_res, 200, 0.95, [1.01,3,-1,1])
c2 = country('count2', all, 200, 0.80, [1.01,3,-1,1])

countries = [c1, c2]

Execute(update_world, 60)

for c in countries:
	c.plot_cases_vs_time()

