import os
import subprocess
import numpy as np
from pprint import pprint
from itertools import zip_longest
from difflib import SequenceMatcher
from colors import red, blue, green, yellow

INPUT_PATH = '/Users/tylerchurchill/Desktop/solitaire-ai/solver/Solvitaire/output.txt'
TRANSLATOR = {
	"kEnd"     : 0,
	"RevealA♠" : 1,
	"Reveal2♠" : 2,
	"Reveal3♠" : 3,
	"Reveal4♠" : 4,
	"Reveal5♠" : 5,
	"Reveal6♠" : 6,
	"Reveal7♠" : 7,
	"Reveal8♠" : 8,
	"Reveal9♠" : 9,
	"RevealT♠" : 10,
	"RevealJ♠" : 11,
	"RevealQ♠" : 12,
	"RevealK♠" : 13,
	"RevealA♥" : 14,
	"Reveal2♥" : 15,
	"Reveal3♥" : 16,
	"Reveal4♥" : 17,
	"Reveal5♥" : 18,
	"Reveal6♥" : 19,
	"Reveal7♥" : 20,
	"Reveal8♥" : 21,
	"Reveal9♥" : 22,
	"RevealT♥" : 23,
	"RevealJ♥" : 24,
	"RevealQ♥" : 25,
	"RevealK♥" : 26,
	"RevealA♣" : 27,
	"Reveal2♣" : 28,
	"Reveal3♣" : 29,
	"Reveal4♣" : 30,
	"Reveal5♣" : 31,
	"Reveal6♣" : 32,
	"Reveal7♣" : 33,
	"Reveal8♣" : 34,
	"Reveal9♣" : 35,
	"RevealT♣" : 36,
	"RevealJ♣" : 37,
	"RevealQ♣" : 38,
	"RevealK♣" : 39,
	"RevealA♦" : 40,
	"Reveal2♦" : 41,
	"Reveal3♦" : 42,
	"Reveal4♦" : 43,
	"Reveal5♦" : 44,
	"Reveal6♦" : 45,
	"Reveal7♦" : 46,
	"Reveal8♦" : 47,
	"Reveal9♦" : 48,
	"RevealT♦" : 49,
	"RevealJ♦" : 50,
	"RevealQ♦" : 51,
	"RevealK♦" : 52,
	"2♠ ← 3♠"  : 53,
	"2♠ ← A♥"  : 54,
	"2♠ ← A♦"  : 55,
	"3♠ ← 4♠"  : 56,
	"3♠ ← 2♥"  : 57,
	"3♠ ← 2♦"  : 58,
	"4♠ ← 5♠"  : 59,
	"4♠ ← 3♥"  : 60,
	"4♠ ← 3♦"  : 61,
	"5♠ ← 6♠"  : 62,
	"5♠ ← 4♥"  : 63,
	"5♠ ← 4♦"  : 64,
	"6♠ ← 7♠"  : 65,
	"6♠ ← 5♥"  : 66,
	"6♠ ← 5♦"  : 67,
	"7♠ ← 8♠"  : 68,
	"7♠ ← 6♥"  : 69,
	"7♠ ← 6♦"  : 70,
	"8♠ ← 9♠"  : 71,
	"8♠ ← 7♥"  : 72,
	"8♠ ← 7♦"  : 73,
	"9♠ ← T♠"  : 74,
	"9♠ ← 8♥"  : 75,
	"9♠ ← 8♦"  : 76,
	"T♠ ← J♠"  : 77,
	"T♠ ← 9♥"  : 78,
	"T♠ ← 9♦"  : 79,
	"J♠ ← Q♠"  : 80,
	"J♠ ← T♥"  : 81,
	"J♠ ← T♦"  : 82,
	"Q♠ ← K♠"  : 83,
	"Q♠ ← J♥"  : 84,
	"Q♠ ← J♦"  : 85,
	"2♥ ← 3♥"  : 86,
	"2♥ ← A♠"  : 87,
	"2♥ ← A♣"  : 88,
	"3♥ ← 4♥"  : 89,
	"3♥ ← 2♠"  : 90,
	"3♥ ← 2♣"  : 91,
	"4♥ ← 5♥"  : 92,
	"4♥ ← 3♠"  : 93,
	"4♥ ← 3♣"  : 94,
	"5♥ ← 6♥"  : 95,
	"5♥ ← 4♠"  : 96,
	"5♥ ← 4♣"  : 97,
	"6♥ ← 7♥"  : 98,
	"6♥ ← 5♠"  : 99,
	"6♥ ← 5♣"  : 100,
	"7♥ ← 8♥"  : 101,
	"7♥ ← 6♠"  : 102,
	"7♥ ← 6♣"  : 103,
	"8♥ ← 9♥"  : 104,
	"8♥ ← 7♠"  : 105,
	"8♥ ← 7♣"  : 106,
	"9♥ ← T♥"  : 107,
	"9♥ ← 8♠"  : 108,
	"9♥ ← 8♣"  : 109,
	"T♥ ← J♥"  : 110,
	"T♥ ← 9♠"  : 111,
	"T♥ ← 9♣"  : 112,
	"J♥ ← Q♥"  : 113,
	"J♥ ← T♠"  : 114,
	"J♥ ← T♣"  : 115,
	"Q♥ ← K♥"  : 116,
	"Q♥ ← J♠"  : 117,
	"Q♥ ← J♣"  : 118,
	"2♣ ← 3♣"  : 119,
	"2♣ ← A♥"  : 120,
	"2♣ ← A♦"  : 121,
	"3♣ ← 4♣"  : 122,
	"3♣ ← 2♥"  : 123,
	"3♣ ← 2♦"  : 124,
	"4♣ ← 5♣"  : 125,
	"4♣ ← 3♥"  : 126,
	"4♣ ← 3♦"  : 127,
	"5♣ ← 6♣"  : 128,
	"5♣ ← 4♥"  : 129,
	"5♣ ← 4♦"  : 130,
	"6♣ ← 7♣"  : 131,
	"6♣ ← 5♥"  : 132,
	"6♣ ← 5♦"  : 133,
	"7♣ ← 8♣"  : 134,
	"7♣ ← 6♥"  : 135,
	"7♣ ← 6♦"  : 136,
	"8♣ ← 9♣"  : 137,
	"8♣ ← 7♥"  : 138,
	"8♣ ← 7♦"  : 139,
	"9♣ ← T♣"  : 140,
	"9♣ ← 8♥"  : 141,
	"9♣ ← 8♦"  : 142,
	"T♣ ← J♣"  : 143,
	"T♣ ← 9♥"  : 144,
	"T♣ ← 9♦"  : 145,
	"J♣ ← Q♣"  : 146,
	"J♣ ← T♥"  : 147,
	"J♣ ← T♦"  : 148,
	"Q♣ ← K♣"  : 149,
	"Q♣ ← J♥"  : 150,
	"Q♣ ← J♦"  : 151,
	"2♦ ← 3♦"  : 152,
	"2♦ ← A♠"  : 153,
	"2♦ ← A♣"  : 154,
	"3♦ ← 4♦"  : 155,
	"3♦ ← 2♠"  : 156,
	"3♦ ← 2♣"  : 157,
	"4♦ ← 5♦"  : 158,
	"4♦ ← 3♠"  : 159,
	"4♦ ← 3♣"  : 160,
	"5♦ ← 6♦"  : 161,
	"5♦ ← 4♠"  : 162,
	"5♦ ← 4♣"  : 163,
	"6♦ ← 7♦"  : 164,
	"6♦ ← 5♠"  : 165,
	"6♦ ← 5♣"  : 166,
	"7♦ ← 8♦"  : 167,
	"7♦ ← 6♠"  : 168,
	"7♦ ← 6♣"  : 169,
	"8♦ ← 9♦"  : 170,
	"8♦ ← 7♠"  : 171,
	"8♦ ← 7♣"  : 172,
	"9♦ ← T♦"  : 173,
	"9♦ ← 8♠"  : 174,
	"9♦ ← 8♣"  : 175,
	"T♦ ← J♦"  : 176,
	"T♦ ← 9♠"  : 177,
	"T♦ ← 9♣"  : 178,
	"J♦ ← Q♦"  : 179,
	"J♦ ← T♠"  : 180,
	"J♦ ← T♣"  : 181,
	"Q♦ ← K♦"  : 182,
	"Q♦ ← J♠"  : 183,
	"Q♦ ← J♣"  : 184,
	"♠ ← A♠"   : 185,
	"♥ ← A♥"   : 186,
	"♣ ← A♣"   : 187,
	"♦ ← A♦"   : 188,
	"🂿 ← K♠"   : 189,
	"🂿 ← K♥"   : 190,
	"🂿 ← K♣"   : 191,
	"🂿 ← K♦"   : 192,
	"A♠ ← 2♠"  : 193,
	"A♥ ← 2♥"  : 194,
	"A♣ ← 2♣"  : 195,
	"A♦ ← 2♦"  : 196,
	"K♠ ← Q♥"  : 197,
	"K♠ ← Q♦"  : 198,
	"K♥ ← Q♠"  : 199,
	"K♥ ← Q♣"  : 200,
	"K♣ ← Q♥"  : 201,
	"K♣ ← Q♦"  : 202,
	"K♦ ← Q♠"  : 203,
	"K♦ ← Q♣"  : 204,
}

def trim_experience(experience):
	_, _, experience = experience.partition('Solution:\n')
	experience, _, _ = experience.rpartition('Solution Type:')
	return experience

def split_experience(trimmed_experience):
	states = trimmed_experience.split('===================================')
	return states

def process_state(state):
	_, foundations, tableaus, waste = state.split('--- ')
	foundations = process_foundations(foundations)
	tableaus    = process_tableaus(tableaus)
	waste       = process_waste(waste)
	return {
		'foundations' : foundations,
		'tableaus' : tableaus,
		'waste' : waste,
	}

def process_foundations(foundations):
	_, _, foundations = foundations.partition('Foundations ---------\n')
	foundations = foundations.strip().split('\t')
	new_foundations = []
	for f, default in zip(foundations, ('♣', '♥', '♠', '♦')):
		if f == '[]':
			new_foundations.append(default)
		else:
			new_foundations.append(f)
	clubs, hearts, spades, diamonds = new_foundations
	foundations = [spades, hearts, clubs, diamonds]
	return foundations

def process_tableaus(tableaus):
	_, _, tableaus_str = tableaus.partition('Tableau Piles -------\n')
	tableaus = tableaus_str.strip().splitlines()
	tableaus = [line.rstrip().split('\t') for line in tableaus]
	tableaus = list(zip_longest(*tableaus, fillvalue=''))
	tableaus = [tuple(filter(lambda n: n != '', t)) for t in  tableaus]
	return tableaus

def process_waste(waste):
	_, _, waste = waste.partition('Stock | Waste -------\n')
	waste = waste.strip().splitlines()
	waste = [line.rstrip().split('\t') for line in waste]
	waste, drawn = list(zip_longest(*waste, fillvalue='[]'))
	waste = list(waste)
	drawn = [card for card in drawn if card != '[]']
	try:
		waste.extend(list(reversed(drawn)))
	except TypeError:
		pass
	return waste

def get_initial_moves(initial_state):
	moves = []
	for t in initial_state['tableaus']:
		moves.append(f'Reveal{t[-1]}')
	for w in initial_state['waste']:
		moves.append(f'Reveal{w}')
	moves = [format_move(m) for m in moves]
	return moves

def diff_states(state, next_state):
	player_move = None
	reveal_move = None

	state_sequences      = [state['foundations']]
	next_state_sequences = [next_state['foundations']]

	for t in state['tableaus']:
		state_sequences.append(t)

	for t in next_state['tableaus']:
		next_state_sequences.append(t)

	for pile, next_pile in zip(state_sequences, next_state_sequences):
		s = SequenceMatcher(None, pile, next_pile)
		for tag, i1, i2, j1, j2 in s.get_opcodes():
			if tag != 'equal' and pile != next_pile:
				# print(blue('\n{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, pile[i1:i2], next_pile[j1:j2])))

				before = pile[i1:i2]
				after  = next_pile[j1:j2]

				# print('pile      :', pile)
				# print('next_pile :', next_pile)
				# print('before :', before)
				# print('after  :', after)

				# Handle moves to empty tableaus or foundations
				if len(pile) == len(next_pile) and after[0] != '[]':
					player_move = f'{before[0]} <- {after[0]}'

				# Handle normal moves
				if len(pile) < len(next_pile):
					if len(pile) == 1 and pile[0] == '[]':
						player_move = f'[] <- {next_pile[0]}'
					else:
						player_move = f'{pile[-1]} <- {next_pile[len(pile)]}'

				# Handle reveal moves
				if len(pile) > len(next_pile):
					for b, a in zip(before, after):
						if b == '##' and a != b:
							reveal_move = f'Reveal{a}'

	executed_moves = [player_move]
	if reveal_move:
		executed_moves.append(reveal_move)
	executed_moves = [format_move(m) for m in executed_moves]

	# print(green(player_move))
	# print(green(reveal_move))

	return executed_moves

def format_move(move_str):
	translator = {
		'S'  : '♠',
		'H'  : '♥',
		'C'  : '♣',
		'D'  : '♦',
		'[]' : '🂿',
		'<-' : '←',
		'10' : 'T'
	}
	for old, new in translator.items():
		move_str = move_str.replace(old, new)
	return move_str

def process_experience(input_path=INPUT_PATH):
	with open(input_path, 'r') as experience_file:
		experience = experience_file.read()

	experience = trim_experience(experience)
	states     = split_experience(experience)
	moves      = get_initial_moves(process_state(states[0]))

	for i in range(len(states) - 1):
		try:
			state      = process_state(states[i + 0])
			next_state = process_state(states[i + 1])
			chosen_moves = diff_states(state, next_state)
		except (IndexError, ValueError):
			chosen_moves = ['kEnd']
		moves.extend(chosen_moves)

	return moves

def generate_experiences(num_samples=1):

	for i in range(num_samples):
		seed = np.random.randint(0, 1e5)
		file = f'./experiences/seed_{seed}.txt'

		subprocess.run(
			['./enter-container.sh',
			 './solvitaire',
			 '--type', 'klondike',
			 '--random', str(seed),
			 '--timeout', '50000',
			 '>>', file],
			cwd='./Solvitaire'
		)

		experience_path = f'./Solvitaire/experiences/seed_{seed}.txt'
		with open(experience_path) as exp_file:
			first_line = exp_file.readline()
			if 'Deal:' in first_line:
				print(red(f'Removing unsolvable file `{experience_path}`'))
				os.remove(experience_path)


if __name__ == '__main__':
	generate_experiences(10)


