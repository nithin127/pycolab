import numpy as np
from pycolab.examples.custom_game import main as get_demo

NUM_DEMOS = 1
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



import curses

def visualise_demo(observation):
	curses.wrapper(visualise)


def visualise(screen):
	self._init_colour()
    curses.curs_set(0)  # We don't need to see the cursor.
    if self._delay is None:
      screen.timeout(-1)  # Blocking reads
    else:
      screen.timeout(self._delay)  # Nonblocking (if 0) or timing-out reads

    # Create the curses window for the log display
    rows, cols = screen.getmaxyx()
    console = curses.newwin(rows // 2, cols, rows - (rows // 2), 0)

    # By default, the log display window is hidden
    paint_console = False

    def crop_and_repaint(observation):
      # Helper for game display: applies all croppers to the observation, then
      # repaints the cropped subwindows. Since the same repainter is used for
      # all subwindows, and since repainters "own" what they return and are
      # allowed to overwrite it, we copy repainted observations when we have
      # multiple subwindows.
      observations = [cropper.crop(observation) for cropper in self._croppers]
      if self._repainter:
        if len(observations) == 1:
          return [self._repainter(observations[0])]
        else:
          return [copy.deepcopy(self._repainter(obs)) for obs in observations]
      else:
        return observations

    # Kick off the game---get first observation, crop and repaint as needed,
    # initialise our total return, and display the first frame.
    observation, reward, _ = self._game.its_showtime()
    observations = crop_and_repaint(observation)
    self._total_return = reward
    self._display(
        screen, observations, self._total_return, elapsed=datetime.timedelta())

     while not self._game.game_over:
      # Wait (or not, depending) for user input, and convert it to an action.
      # Unrecognised keycodes cause the game display to repaint (updating the
      # elapsed time clock and potentially showing/hiding/updating the log
      # message display) but don't trigger a call to the game engine's play()
      # method. Note that the timeout "keycode" -1 is treated the same as any
      # other keycode here.

      # Load the agent policy here:
      action = self.agent.agent_network(observation, self._action_list)
      time.sleep(0.2)
      observation, reward, _ = self._game.play(action)
      observations = crop_and_repaint(observation)
      if self._total_return is None:
        self._total_return = reward
      elif reward is not None:
        self._total_return += reward

      # Update the game display, regardless of whether we've called the game's
      # play() method.
      elapsed = datetime.datetime.now() - self._start_time
      self._display(screen, observations, self._total_return, elapsed)

      # Update game console message buffer with new messages from the game.
      self._update_game_console(
          plab_logging.consume(self._game.the_plot), console, paint_console)

      # Show the screen to the user.
      curses.doupdate()



def main(vis = False):
	mazes = []
	demos = []
	for _ in range(NUM_DEMOS):
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
		import ipdb; ipdb.set_trace()
		if vis:
			visualise_demo(demo)
		mazes.append(maze)
		demos.append(demo)



if __name__ == "__main__":
	main()




