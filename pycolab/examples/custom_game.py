# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A scrolling maze to explore. Collect all of the coins!

The scrolling mechanism used by this example is a bit old-fashioned. For a
recommended simpler, more modern approach to scrolling in games with finite
worlds, have a look at `better_scrolly_maze.py`. On the other hand, if you have
a game with an "infinite" map (for example, a maze that generates itself "on
the fly" as the agent walks through it), then a mechanism using the scrolling
protocol (as the game entities in this game do) is worth investigating.

Command-line usage: `scrolly_maze.py <level>`, where `<level>` is an optional
integer argument selecting Scrolly Maze levels 0, 1, or 2.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys

from pycolab import ascii_art
from pycolab import agent_ui
from pycolab.prefab_parts import drapes as prefab_drapes
from pycolab.prefab_parts import sprites as prefab_sprites


# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the patroller sprites
    # 'a', 'b', and 'c'. I guess if you really don't want them in your maze, you
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': patroller A.
    #     '@': collectable coins.           'b': patroller B.
    #     'P': player starting location.    'c': patroller C.
    #     ' ': boring old maze floor.       '+': initial board top-left corner.
    #
    # Don't forget to specify the initial board scrolling position with '+', and
    # take care that it won't cause the board to extend beyond the maze.
    # Remember also to update the MAZES_WHAT_LIES_BENEATH array whenever you add
    # a new maze.

    # Maze #0:
    # This is the minimum possible width and height of the maze,
    # that doesn't give an error: with Scrolly.Maze
    # Maybe think of implementing something low-key if this gives a problem

     ['+#############################',
     '#                            #',
     '#     #       #    #     c   #',
     '#     #       #    #         #',
     '# %   #  ######    #####  ####',
     '#        @   #               #',
     '#            #####      *    #',
     '#   P                 ########',
     '#                            #',
     '##############################'],

    # Maze #1:
     ['##############################',
     '#       #                #   #',
     '#   #   #####   #####    #   #',
     '#   #   #       #   #        #',
     '+   #####   #####   ##########',
     '#   #       #                #',
     '#   #   #   # c ##########   #',
     '#   #   #   #       #        #',
     '#   #   #####   ######   #####',
     '#       # @  P       #   #   #',
     '#   #####   #####  * #   #   #',
     '#   #       #   #    #   #   #',
     '#   #   #####   # %  #   #   #',
     '#   #   #       #        #   #',
     '#   #   #   #   #    #####   #',
     '#   #   #   #   #        #   #',
     '#   #   #   #   #    #   #   #',
     '#           #        #       #',
     '##############################'],

    # Maze #2:
    ['##############################',
     '#       #       c        #   #',
     '#   #   #####   #####    #   #',
     '# @ #   #       #   #        #',
     '#   #####   #####   ##########',
     '#   #       #     @     @    #',
     '# @ #   #   # @ ##########   #',
     '#   #   #   #     @ #  @   @ #',
     '+   #   ##############   #####',
     '# @     # @  P*   @  #   #   #',
     '#   #####   #####    # @ #   #',
     '#   # @   @ #   #    #   #   #',
     '# @ #   #####   # @  #   #   #',
     '#   #   #       #      @ #   #',
     '#   # @ #   #   ##########   #',
     '# @ #   #   #   #        #   #',
     '#   #   #   #   #    #   #   #',
     '#     @     #        #       #',
     '##############################']]

# pylint: enable=line-too-long

MAZES_WHAT_LIES_BENEATH = [
    # What lies below '+' characters in MAZES_ART?
    # Unlike the what_lies_beneath argument to ascii_art_to_game, only single
    # characters are supported here for the time being.

    '#',  # Maze #0
    '#',  # Maze #1
    '#',  # Maze #2
]


STAR_ART = ['  .           .          .    ',
            '         .       .        .   ',
            '        .          .         .',
            '  .    .    .           .     ',
            '.           .          .   . .',
            '         .         .         .',
            '   .                 .        ',
            '           . .          .     ',
            '    .            .          . ',
            '  .      .              .  .  ']


TERMINATION_SEQUENCE = ['c', '*', '%', '@']
CURRENT_SEQUENCE = []
# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Inky blackness of SPAAAACE
             '.': (949, 929, 999),  # These stars are full of lithium
             '@': (999, 862, 110),  # Shimmering golden coins
             '#': (764, 0, 999),    # Walls of the SPACE MAZE
             '*': (987, 623, 145),  # Diamond
             '%': (987, 623, 145),  # Tree
             'c': (0, 999, 999),    # This is you, the player
             'P': (0, 999, 999),}    # This is you, the player

COLOUR_BG = {'.': (0, 0, 0),        # Around the stars, inky blackness etc.
             '@': (0, 0, 0),
             '*': (0, 0, 0),
             '%': (0, 0, 0),
             'c': (0, 0, 0)}

Z_ORDER = '%@*c#P'


def make_game(level):
  """Builds and returns a Scrolly Maze game for the selected level."""
  # A helper object that helps us with Scrolly-related setup paperwork.
  scrolly_info = prefab_drapes.Scrolly.PatternInfo(
      MAZES_ART[level], STAR_ART,
      board_northwest_corner_mark='+',
      what_lies_beneath=MAZES_WHAT_LIES_BENEATH[level])

  walls_kwargs = scrolly_info.kwargs('#')
  coins_kwargs = scrolly_info.kwargs('@')
  cow_kwargs = scrolly_info.kwargs('c')
  diamond_kwargs = scrolly_info.kwargs('*')
  tree_kwargs = scrolly_info.kwargs('%')
  player_position = scrolly_info.virtual_position('P')

  return ascii_art.ascii_art_to_game(
      STAR_ART, what_lies_beneath=' ',
      sprites={
          'P': ascii_art.Partial(PlayerSprite, player_position)},
      drapes={
          '#': ascii_art.Partial(MazeDrape, **walls_kwargs),
          '*': ascii_art.Partial(DiamondDrape, **diamond_kwargs),
          'c': ascii_art.Partial(CowDrape, **cow_kwargs),
          '%': ascii_art.Partial(TreeDrape, **tree_kwargs),
          '@': ascii_art.Partial(CashDrape, **coins_kwargs)},
      # The base Backdrop class will do for a backdrop that just sits there.
      # In accordance with best practices, the one egocentric MazeWalker (the
      # player) is in a separate and later update group from all of the
      # pycolab entities that control non-traversable characters.
      update_schedule=[['#'], ['P'], ['*','c','@','%']],
      z_order=Z_ORDER)


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer.

  This egocentric `Sprite` requires no logic beyond tying actions to
  `MazeWalker` motion action helper methods, which keep the player from walking
  on top of obstacles.
  """

  def __init__(self, corner, position, character, virtual_position):
    """Constructor: player is egocentric and can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, egocentric_scroller=True, impassable='#')
    self._teleport(virtual_position)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions in range(4,9):  # do nothing?
      self._stay(board, the_plot)

    # Make sure that the player is obstructed if there's some stuff happening



class MazeDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling the maze scenery.

  This `Drape` requires no logic beyond tying actions to `Scrolly` motion
  action helper methods. Our job as programmers is to make certain that the
  actions we use have the same meaning between all `Sprite`s and `Drape`s in
  the same scrolling group (see `protocols/scrolling.py`).
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions in range(4,9):  # is the player doing nothing?
      pass
      #self._stay(the_plot)



def general_update(player_pattern_position, whole_pattern, actions, the_plot, char = "@"):
    # If the player has reached a coin, credit one reward and remove the coin
    # from the scrolling pattern. If the player has obtained all coins, quit!
    
    px = 0
    py = 0
    
    if actions == 5:    # is the player picking upward?
      py = -1
    elif actions == 6:  # is the player picking downward?
      py = 1
    elif actions == 7:  # is the player picking leftward?
      px = -1
    elif actions == 8:  # is the player picking rightward?    
      px = 1
    elif actions == 9:  # does the player want to quit?
      the_plot.terminate_episode()

    if whole_pattern[(player_pattern_position[0] + py, player_pattern_position[1] + px)]:
      the_plot.log('{} collected at {}!'.format(char, player_pattern_position))
      if not (px + py == 0):
        CURRENT_SEQUENCE.append('p' + char)
        the_plot.add_reward(100)
        whole_pattern[(player_pattern_position[0] + py, player_pattern_position[1] + px)] = False
      else:
        CURRENT_SEQUENCE.append('v' + char)
      print(CURRENT_SEQUENCE)
      if len(CURRENT_SEQUENCE) == len(TERMINATION_SEQUENCE):
        if CURRENT_SEQUENCE == TERMINATION_SEQUENCE:
          the_plot.add_reward(100)
          the_plot.log('Successfully executed sequence')
          the_plot.terminate_episode()



class CashDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling all of the coins.

  This `Drape` ties actions to `Scrolly` motion action helper methods, and once
  again we take care to map the same actions to the same methods. A little
  extra logic updates the scrolling pattern for when the player touches the
  coin, credits reward, and handles game termination.
  """
  def update(self, actions, board, layers, backdrop, things, the_plot):
    player_pattern_position = self.pattern_position_prescroll(
        things['P'].position, the_plot)

    general_update(player_pattern_position, self.whole_pattern, actions, the_plot, "@")

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions in range(4,9):  # is the player doing nothing?
      self._stay(the_plot)
    elif actions == 9:  # does the player want to quit?
      the_plot.terminate_episode()



class DiamondDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling all of the coins.

  This `Drape` ties actions to `Scrolly` motion action helper methods, and once
  again we take care to map the same actions to the same methods. A little
  extra logic updates the scrolling pattern for when the player touches the
  coin, credits reward, and handles game termination.
  """
  def update(self, actions, board, layers, backdrop, things, the_plot):
    del board, layers, backdrop
    player_pattern_position = self.pattern_position_prescroll(
        things['P'].position, the_plot)

    general_update(player_pattern_position, self.whole_pattern, actions, the_plot, "*")

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions in range(4,9):  # is the player doing nothing?
      self._stay(the_plot)
    elif actions == 9:  # does the player want to quit?
      the_plot.terminate_episode()



class CowDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling all of the coins.

  This `Drape` ties actions to `Scrolly` motion action helper methods, and once
  again we take care to map the same actions to the same methods. A little
  extra logic updates the scrolling pattern for when the player touches the
  coin, credits reward, and handles game termination.
  """
  def update(self, actions, board, layers, backdrop, things, the_plot):
    del board, layers, backdrop
    player_pattern_position = self.pattern_position_prescroll(
        things['P'].position, the_plot)

    general_update(player_pattern_position, self.whole_pattern, actions, the_plot, "c")

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions in range(4,9):  # is the player doing nothing?
      self._stay(the_plot)
    elif actions == 9:  # does the player want to quit?
      the_plot.terminate_episode()



class TreeDrape(prefab_drapes.Scrolly):
  """A scrolling `Drape` handling all of the coins.

  This `Drape` ties actions to `Scrolly` motion action helper methods, and once
  again we take care to map the same actions to the same methods. A little
  extra logic updates the scrolling pattern for when the player touches the
  coin, credits reward, and handles game termination.
  """
  def update(self, actions, board, layers, backdrop, things, the_plot):
    del board, layers, backdrop
    player_pattern_position = self.pattern_position_prescroll(
        things['P'].position, the_plot)

    general_update(player_pattern_position, self.whole_pattern, actions, the_plot, "%")

    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions in range(4,9):  # is the player doing nothing?
      self._stay(the_plot)
    elif actions == 9:  # does the player want to quit?
      the_plot.terminate_episode()



def main(argv=()):
  # Build a Scrolly Maze game.
  game = make_game(int(argv[1]) if len(argv) > 1 else 0)

  # Make a CursesUi to play it with.
  ui = agent_ui.AgentUi(
        keys_to_actions={
          # Basic movement.
          curses.KEY_UP: 0,
          curses.KEY_DOWN: 1,
          curses.KEY_LEFT: 2,
          curses.KEY_RIGHT: 3,
          # Shoot aperture gun.
          'w': 5,
          'a': 7,
          's': 6,
          'd': 8,
          # Quit game.
          'q': 9,
          'Q': 9,}, 
        delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
