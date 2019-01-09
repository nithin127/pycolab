import numpy as np
from pycolab.examples.custom_game import main as get_demo
# from agent_ui import AgentUi

NUM_DEMOS = 1000
MIN_INSTR = 3
MAX_INSTR = 5
NUM_OBJECTS = 6

objects = '$H&@J/*c!d'

MAZE_ART = \
    ['+#############################',
     '#                            #',
     '#                            #',
     '#                  #         #',
     '#          P       #         #',
     '#                  #         #',
     '#      #######     #         #',
     '#                  #         #',
     '#                            #',
     '##############################']

# len(POSSIBLE_OBJECT_POSITIONS) should be at least NUM_OBJECTS
POSSIBLE_OBJECT_POSITIONS = [(3,4), (8,27), (6,3), (4,20), (2,15), (7,3)] 


#------------------------------ MAIN LOGIC ------------------------------#


def main(vis = False):
	mazes = []
	demos = []
	for nd in range(NUM_DEMOS):
		try:
			chars = []
			instr = []

			for _ in range(NUM_OBJECTS):
			    chars.append(objects[np.random.choice(len(objects))])

			num_instr = np.random.choice(MAX_INSTR - MIN_INSTR + 1) + MIN_INSTR
			char_indices = np.random.choice(len(chars), num_instr, replace=False)
			np.random.shuffle(POSSIBLE_OBJECT_POSITIONS)
			maze = MAZE_ART.copy()

			for i, ind in enumerate(char_indices):
				instr.append(np.random.choice(('v','p')) + chars[ind])

			for i, c in enumerate(chars):
				ci, cj = POSSIBLE_OBJECT_POSITIONS[i]
				maze[ci] = maze[ci][:cj] + c + maze[ci][cj+1:]

			# Get the demonstration
			demo = get_demo(['', maze, instr, True])
			if vis:
				visualise_demo(demo)
			mazes.append(maze)
			demos.append({"observations": demo[0], "actions": demo[1], "targets": demo[2], })
			print(nd)

		except:
			# Some shit going on here, figure out later
			pass

	import ipdb; ipdb.set_trace()	
	import pickle
	pickle.dump({"mazes": mazes, "demos": demos}, open("mazes_and_demos.pk", "wb"))
	return mazes, demos


if __name__ == "__main__":
	main()




