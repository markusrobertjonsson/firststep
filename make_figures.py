import numpy as np
import itertools
from random import random, choice
from math import exp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 18


NOISE = 'N'


def progress_print(string, **kwargs):
    print(string, **kwargs)


def sliding_average(values, sample_size):
    out = [None] * len(values)
    for i, val in enumerate(values):
        if i < sample_size:
            out[i] = sum(values[0: (i + 1)]) / (i + 1)
        else:
            out[i] = sum(values[(i - sample_size): i]) / sample_size
    return out


class World():
    """
    A class that can return sequences of stimuli.
    """

    def __init__(self, world_parm):
        self.world_parm = world_parm
        self._make_sequence_templates()
        self._make_test_sequences()
        self.print_info()

    def _make_sequence_templates(self):
        self.sequence_templates = []
        n_noise_stimuli = self.world_parm.get('n_noise_stimuli')
        stimulus_pool = itertools.count(start=1, step=1)

        world_type = self.world_parm.get('world_type')

        if world_type == 'sequence':
            a = next(stimulus_pool)
            b = next(stimulus_pool)

            t = [[a, b], [b, a]]
            t += [[a, NOISE] for i in range(50)]
            t += [[b, NOISE] for i in range(50)]
            t += [[NOISE, a] for i in range(50)]
            t += [[NOISE, b] for i in range(50)]
            self._append_sequence_template(t,
                                           [True, False] + [None] * 200)

            n_informative_stimuli = next(stimulus_pool) - 1

        elif world_type == 'vector':
            n_inf_sequences = self.world_parm.get('n_inf_sequences')

            # Whether or not to reuse informative stimuli at one level in the other
            reuse_informative_stimuli = self.world_parm.get('reuse_informative_stimuli')

            n_informative_stimuli = self.world_parm.get('n_inf_stimuli')
            if n_informative_stimuli == 'free':
                if reuse_informative_stimuli:
                    n_informative_stimuli = max(n_inf_sequences)
                else:
                    n_informative_stimuli = sum(n_inf_sequences)

            stimulus_pool = itertools.cycle(range(1, n_informative_stimuli + 1))
            seq_length = len(n_inf_sequences)
            for index, n in enumerate(n_inf_sequences):
                if reuse_informative_stimuli:
                    # stimulus_pool = iter(PosIntIter())
                    stimulus_pool = itertools.cycle(range(1, n_informative_stimuli + 1))
                for i in range(n):
                    seq = [NOISE] * seq_length
                    seq[index] = next(stimulus_pool)
                    # is_rewarded = ((i % 2) == 1)
                    is_rewarded = ((seq[index] % 2) == 1)
                    self._append_sequence_template([seq], [is_rewarded])

        elif world_type == 'parameter_study_mixed':
            a = next(stimulus_pool)
            b = next(stimulus_pool)
            c = next(stimulus_pool)
            d = next(stimulus_pool)

            # e = next(stimulus_pool)
            # f = next(stimulus_pool)
            # g = next(stimulus_pool)
            # h = next(stimulus_pool)
            # i = next(stimulus_pool)
            # j = next(stimulus_pool)

            # Only for printing
            self._append_sequence_template([[a, b, NOISE, NOISE],
                                            [NOISE, a, b, NOISE],
                                            [NOISE, NOISE, a, b],
                                            [b, a, NOISE, NOISE],
                                            [NOISE, b, a, NOISE],
                                            [NOISE, NOISE, b, a],
                                            [a, NOISE, NOISE, NOISE],
                                            [b, NOISE, NOISE, NOISE],
                                            [NOISE, a, NOISE, NOISE],
                                            [NOISE, b, NOISE, NOISE],
                                            [NOISE, NOISE, a, NOISE],
                                            [NOISE, NOISE, b, NOISE],
                                            [NOISE, NOISE, NOISE, a],
                                            [NOISE, NOISE, NOISE, b]],
                                           [True, True, True, False, False, False,
                                            True, False, True, False, True, False, True, False])

            n_informative_stimuli = next(stimulus_pool) - 1
            self.culture_sequence_templates = [SequenceTemplate([a, b, NOISE, NOISE], True),
                                               SequenceTemplate([NOISE, a, b, NOISE], True),
                                               SequenceTemplate([NOISE, NOISE, a, b], True),
                                               SequenceTemplate([b, a, NOISE, NOISE], False),
                                               SequenceTemplate([NOISE, b, a, NOISE], False),
                                               SequenceTemplate([NOISE, NOISE, b, a], False)]
            self.culture_sequences_noise = [SequenceTemplate([a, NOISE, NOISE, NOISE], None),
                                            SequenceTemplate([NOISE, a, NOISE, NOISE], None),
                                            SequenceTemplate([NOISE, NOISE, a, NOISE], None),
                                            SequenceTemplate([NOISE, NOISE, NOISE, a], None),
                                            SequenceTemplate([b, NOISE, NOISE, NOISE], None),
                                            SequenceTemplate([NOISE, b, NOISE, NOISE], None),
                                            SequenceTemplate([NOISE, NOISE, b, NOISE], None),
                                            SequenceTemplate([NOISE, NOISE, NOISE, b], None)]
            self.nature_sequence_templates = [SequenceTemplate([c, NOISE, NOISE, NOISE], True),
                                              SequenceTemplate([d, NOISE, NOISE, NOISE], False),
                                              SequenceTemplate([NOISE, c, NOISE, NOISE], True),
                                              SequenceTemplate([NOISE, d, NOISE, NOISE], False),
                                              SequenceTemplate([NOISE, NOISE, c, NOISE], True),
                                              SequenceTemplate([NOISE, NOISE, d, NOISE], False),
                                              SequenceTemplate([NOISE, NOISE, NOISE, c], True),
                                              SequenceTemplate([NOISE, NOISE, NOISE, d], False)]

        self.noise_stimuli = list(range(n_informative_stimuli + 1, n_informative_stimuli + n_noise_stimuli + 2))
        self.n_stimuli = n_informative_stimuli + n_noise_stimuli

    def _append_sequence_template(self, seqs, is_rewardeds):
        assert(len(seqs) == len(is_rewardeds))
        for seq, is_rewarded in zip(seqs, is_rewardeds):
            sequence_template = SequenceTemplate(seq, is_rewarded)
            self.sequence_templates.append(sequence_template)

    def get_next_sequence(self):
        sequence_template = choice(self.sequence_templates)
        sequence = sequence_template.make_sequence(self.noise_stimuli)
        return sequence.sequence, sequence.is_rewarded

    def get_next_sequence_culture_p(self):
        assert(self.world_parm.get('world_type') == 'parameter_study_mixed')
        is_culture_sequence = (random() < self.world_parm.get('p'))
        if is_culture_sequence:
            sequence_template = choice(self.culture_sequence_templates)
        else:
            sequence_template = choice(self.nature_sequence_templates)
        sequence = sequence_template.make_sequence(self.noise_stimuli)
        return sequence.sequence, sequence.is_rewarded, is_culture_sequence

    def get_next_culture_sequence_noise(self):
        assert(self.world_parm.get('world_type') == 'parameter_study_mixed')
        sequence_template = choice(self.culture_sequences_noise)
        # sequence_template = self.culture_sequences_noise[i // 25]
        sequence = sequence_template.make_sequence(self.noise_stimuli)
        return sequence.sequence, sequence.is_rewarded

    def _make_test_sequences(self):
        has_noise = False
        for sequence_template in self.sequence_templates:
            if NOISE in sequence_template.sequence_template:
                has_noise = True
                break
        n_test_sequences = self.world_parm.get('n_test_sequences')
        self.test_sequences = []
        if has_noise:

            world_type = self.world_parm.get('world_type')
            if world_type == 'sequence':
                self.test_sequences.append(self.sequence_templates[0].make_sequence(None))
                self.test_sequences.append(self.sequence_templates[1].make_sequence(None))
            elif world_type == 'parameter_study_mixed':
                p = self.world_parm.get('p')

                # Sequences from the culture world
                n_culture_templates = int(p * n_test_sequences)
                assert(n_culture_templates % 6 == 0), "p times n_test_sequences must be multiple of 6."
                # culture_templates_cycle = itertools.cycle(self.culture_sequence_templates)
                culture_templates_cycle = itertools.cycle(self.culture_sequence_templates)
                # culture_templates_cycle = itertools.cycle([SequenceTemplate([a, b], True),
                #                                            SequenceTemplate([b, a], False)])
                for _ in range(n_culture_templates):
                    culture_template = next(culture_templates_cycle)
                    test_sequence = culture_template.make_sequence(self.noise_stimuli)
                    self.test_sequences.append(test_sequence)

                # Sequences from the nature world
                n_nature_templates = n_test_sequences - n_culture_templates
                assert_str = f"(1-p) times n_test_sequences ({n_nature_templates}) must be multiple of 8."
                assert(n_nature_templates % 8 == 0), assert_str
                nature_templates_cycle = itertools.cycle(self.nature_sequence_templates)
                for _ in range(n_nature_templates):
                    nature_template = next(nature_templates_cycle)
                    test_sequence = nature_template.make_sequence(self.noise_stimuli)
                    self.test_sequences.append(test_sequence)

            else:
                assert(type(n_test_sequences) is int)
                assert(n_test_sequences % len(self.sequence_templates) == 0 and n_test_sequences > 0), "Number of test sequences must be a multiple of the number of template sequences."
                sequence_templates_rewarding = [s for s in self.sequence_templates if s.is_rewarded]
                sequence_templates_nonrewarding = [s for s in self.sequence_templates if not s.is_rewarded]
                assert(len(sequence_templates_rewarding) == len(sequence_templates_nonrewarding))

                sequence_templates_cycle = itertools.cycle(self.sequence_templates)
                for _ in range(n_test_sequences):
                    sequence_template = next(sequence_templates_cycle)
                    test_sequence = sequence_template.make_sequence(self.noise_stimuli)
                    self.test_sequences.append(test_sequence)

        else:
            assert(n_test_sequences == 'all')
            for sequence_template in self.sequence_templates:
                test_sequence = sequence_template.make_sequence([])
                self.test_sequences.append(test_sequence)

    def print_info(self):
        print("\n\nWORLD SUMMARY")
        print("First index is last stimulus, second is previous stimulus, etc.")
        print("Sequence             Is rewarded")
        print("--------             -----------")
        for template in self.sequence_templates:
            to_display = list(template.sequence_template)
            for i in range(len(to_display)):
                if to_display[i] == NOISE:
                    to_display[i] = 0
            print(f"{to_display}         {template.is_rewarded}")
        print(f"Total number of stimuli: {self.n_stimuli}")
        print(f"test sequences ({len(self.test_sequences)}): {[s.sequence for s in self.test_sequences]}")


class SequenceTemplate():
    def __init__(self, sequence_template, is_rewarded):
        self.sequence_template = sequence_template
        self.is_rewarded = is_rewarded

    def make_sequence(self, noise_stimuli):
        sequence = list(self.sequence_template)
        for i, stimulus in enumerate(sequence):
            if stimulus == NOISE:
                sequence[i] = choice(noise_stimuli)
        return Sequence(sequence, self.is_rewarded)

    def match(self, seq, noise_stimuli):
        for s, t in zip(seq, self.sequence_template):
            if t == NOISE:
                is_match = (s in noise_stimuli)
            else:
                is_match = (s == t)
            if not is_match:
                return False
        return True


class Sequence():
    def __init__(self, sequence, is_rewarded):
        self.sequence = sequence
        self.is_rewarded = is_rewarded


# ------------------------------------------------------------
#                          Mechanisms
# ------------------------------------------------------------
class Learner():
    """
    A learning individual.

    Attributes:
        parm (LearnerParameters): The learner parameters.
        sr_value (dict): The stimulus-response values. Keys are stimuli, values are
            lists of length two: [value_not_responding, value_responding].
        representation (dict): The representation of the memory. Keys are stumuli, values are
            intensities.
    """

    def __init__(self, parm):
        self.parm = parm
        self._create_start_sr()
        self._create_start_representation()

    def _create_start_sr(self):
        self.sr_value = dict()

    def _initial_sr(self):
        sr_noresponse = self.parm.get('start_sr_noresponse')
        sr_response = self.parm.get('start_sr_response')
        return [sr_noresponse, sr_response]

    def _create_start_representation(self):
        self.representation = dict()

    def respond_to_sequence(self, seq, is_rewarded):
        self._create_start_representation()
        for stimulus in seq:
            self._update_representation(stimulus)
        self._respond_and_learn(is_rewarded)

    def _respond_and_learn(self, is_rewarded):
        response, u = self._get_response_and_u(is_rewarded)

        # Learn
        v_sum = 0
        alpha = self.parm.get('alpha')
        for s, intensity in self.representation.items():
            v_sum += self.sr_value[s][response] * intensity

        for s, intensity in self.representation.items():
            self.sr_value[s][response] += alpha * (u - v_sum) * intensity

    def _get_response_and_u(self, is_rewarded):
        """Get response and reinforcement value u."""

        # assert (is_rewarded is not None)
        if is_rewarded is None:
            is_rewarded = choice([True, False])

        if random() <= self._p_response():  # Individual is responding
            response = 1
            if is_rewarded:
                u = self.parm.get('value_rewarded_response')
            else:
                u = self.parm.get('value_punished_response')
        else:  # Individual is not responding
            response = 0
            u = self.parm.get('value_no_response')

            # If reinforcement is used also on no-go
            # if is_rewarded:
            #     u = self.parm.get('value_punished_response')
            # else:
            #     u = self.parm.get('value_rewarded_response')

        return response, u

    def _p_response(self):
        beta = self.parm.get('beta')
        support_response = 0
        support_noresponse = 0
        for s, intensity in self.representation.items():
            if s not in self.sr_value:
                self.sr_value[s] = self._initial_sr()
            support_response += self.sr_value[s][1] * intensity
            support_noresponse += self.sr_value[s][0] * intensity
        support_response = exp(beta * support_response)
        support_noresponse = exp(beta * support_noresponse)
        return support_response / (support_response + support_noresponse)

    def p_data(self, test_seq):  # For measuring performance using test sequences
        """Return the probability of response to each stimulus in test_seq."""
        self._create_start_representation()
        test_seq_len = len(test_seq)
        p = [None] * test_seq_len
        for i in range(test_seq_len):
            self._update_representation(test_seq[i])
            p[i] = self._p_response()
        return p


class CurrentStimulusLearner(Learner):
    def __init__(self, parm):
        super().__init__(parm)

    def _update_representation(self, stimulus):
        self.representation = {stimulus: 1}


class TraceLearner(Learner):
    def __init__(self, parm):
        super().__init__(parm)
        self.trace = parm.get('trace')

    def _update_representation(self, stimulus):
        for s in self.representation:
            self.representation[s] *= self.trace
        self.representation[stimulus] = 1


class TimeTaggingLearner(Learner):
    def __init__(self, parm):
        super().__init__(parm)

    def _create_start_sr(self):
        self.sr_value = [dict() for _ in range(self.parm.get('depth'))]

    def _create_start_representation(self):
        self.representation = [dict() for _ in range(self.parm.get('depth'))]

    def _update_representation(self, stimulus):
        for i in reversed(range(1, len(self.representation))):
            self.representation[i] = dict(self.representation[i - 1])
        self.representation[0] = {stimulus: 1}  # self._create_repr_dict(stimulus)

    def _respond_and_learn(self, is_rewarded):
        response, u = self._get_response_and_u(is_rewarded)

        # Learn
        v_sum = 0
        alpha = self.parm.get('alpha')
        for i in range(len(self.representation)):
            for s, intensity in self.representation[i].items():
                if s not in self.sr_value[i]:
                    self.sr_value[i][s] = self._initial_sr()
                v_sum += self.sr_value[i][s][response] * intensity
        for i in range(len(self.representation)):
            for s, intensity in self.representation[i].items():
                self.sr_value[i][s][response] += alpha * (u - v_sum) * intensity

    def _p_response(self):
        beta = self.parm.get('beta')
        support_response = 0
        support_no = 0
        for i in range(len(self.representation)):
            for s, intensity in self.representation[i].items():
                if s not in self.sr_value[i]:
                    self.sr_value[i][s] = self._initial_sr()
                support_response += self.sr_value[i][s][1] * intensity
                support_no += self.sr_value[i][s][0] * intensity
        support_response = exp(beta * support_response)
        support_no = exp(beta * support_no)
        return support_response / (support_response + support_no)


class FlexibleSequenceLearnerOneTag(Learner):
    def __init__(self, parm):
        super().__init__(parm)

    def _consecutive_groups(self):
        groups = list()

        # Add all consecutive stimulus pairs to groups
        for i in range(self.parm.get('depth') - 1):
            s = self.representation[i]
            s_prev = self.representation[i + 1]
            group = (s_prev, s)
            if None not in group:
                groups.append(group)

        # Add the three last stimuli
        if self.parm.get('depth') > 2:
            s0 = self.representation[0]
            s1 = self.representation[1]
            s2 = self.representation[2]
            group = (s0, s1, s2)
            if None not in group:
                groups.append(group)

        # Add the three first stimuli
        if self.parm.get('depth') > 3:
            s1 = self.representation[1]
            s2 = self.representation[2]
            s3 = self.representation[3]
            group = (s1, s2, s3)
            if None not in group:
                groups.append(group)

            # The last four
            s0 = self.representation[0]
            group = (s0, s1, s2, s3)
            if None not in group:
                groups.append(group)

        return groups

    def _create_start_sr(self):
        super()._create_start_sr()
        self.sr_value_groups = dict()

    def _create_start_representation(self):
        self.representation = [None] * self.parm.get('depth')
        self.representation_groups = dict()

    def _update_representation(self, stimulus):
        # Update representation of single stimuli
        for i in reversed(range(1, self.parm.get('depth'))):
            self.representation[i] = self.representation[i - 1]
        self.representation[0] = stimulus

        # Update representation of groups of stimuli
        for group in self._consecutive_groups():
            if group not in self.representation_groups:
                self.representation_groups[group] = 1

            # Update also sr_value_groups
            if group not in self.sr_value_groups:
                self.sr_value_groups[group] = self._initial_sr()

    def _respond_and_learn(self, is_rewarded):
        response, u = self._get_response_and_u(is_rewarded)

        # Learn
        v_sum = 0
        alpha = self.parm.get('alpha')
        for s in self.representation:
            if s not in self.sr_value:
                self.sr_value[s] = self._initial_sr()
            v_sum += self.sr_value[s][response]

        for group in self.representation_groups:
            v_sum += self.sr_value_groups[group][response]

        for s in self.representation:
            self.sr_value[s][response] += alpha * (u - v_sum)

        for group in self.representation_groups:
            self.sr_value_groups[group][response] += alpha * (u - v_sum)

    def _p_response(self):
        beta = self.parm.get('beta')
        support_response = 0
        support_no = 0
        for s in self.representation:
            if s not in self.sr_value:
                self.sr_value[s] = self._initial_sr()
            support_response += self.sr_value[s][1]
            support_no += self.sr_value[s][0]

        for group in self._consecutive_groups():
            support_response += self.sr_value_groups[group][1]
            support_no += self.sr_value_groups[group][0]

        support_response = exp(beta * support_response)
        support_no = exp(beta * support_no)
        return support_response / (support_response + support_no)


class UniqueSequenceLearner(Learner):
    def __init__(self, parm):
        super().__init__(parm)

    def _create_start_sr(self):
        self.sr_value = dict()  # Keys are sequences, values are [no response val, response val]

    def _create_start_representation(self):
        self.representation = (None,) * self.parm.get('depth')

    def _update_representation(self, stimulus):
        repr_list = list(self.representation)
        repr_list.pop()
        repr_list.insert(0, stimulus)
        self.representation = tuple(repr_list)

    def _respond_and_learn(self, is_rewarded):
        response, u = self._get_response_and_u(is_rewarded)

        # Learn
        if self.representation not in self.sr_value:
            self.sr_value[self.representation] = self._initial_sr()
        v_sum = self.sr_value[self.representation][response]
        alpha = self.parm.get('alpha')
        self.sr_value[self.representation][response] += alpha * (u - v_sum)

    def _p_response(self):
        beta = self.parm.get('beta')
        support_response = 0
        support_no = 0
        if self.representation not in self.sr_value:
            self.sr_value[self.representation] = self._initial_sr()

        support_response = self.sr_value[self.representation][1]
        support_no = self.sr_value[self.representation][0]
        support_response = exp(beta * support_response)
        support_no = exp(beta * support_no)
        return support_response / (support_response + support_no)


# ------------------------------------------------------------
#                            Simulation
# ------------------------------------------------------------
class Simulation():
    def __init__(self, world, sim_parm, learner_parm):
        self.world = world
        self.sim_parm = sim_parm
        if sim_parm.get('collect_when') == 'every100':
            self.sim_parm.set(collect_when=int(sim_parm.get('trials') / 100))
        self.learner_parm = learner_parm
        size = sim_parm.get('trials') + 1
        self.data = [0] * size

    def collect_data(self, trial, individual):
        collect_when = self.sim_parm.get('collect_when')
        n_trials = self.sim_parm.get('trials')
        is_last_trial = (trial == n_trials)
        if (collect_when == 'last') and (not is_last_trial):
            return

        if collect_when == 'last' or collect_when == 'each':
            do_collect = True
        elif type(collect_when) is int:
            do_collect = (trial % collect_when == 0)

        if not do_collect:
            self.data[trial] = None
            return

        runs = self.sim_parm.get('runs')
        measure_data = 0
        measure_ndata = 0

        for test_sequence in self.world.test_sequences:
            seq = test_sequence.sequence
            p = individual.p_data(seq)[-1]
            # input(individual.p_data(seq))

            if test_sequence.is_rewarded:
                measure_data += p
            else:
                measure_data += (1 - p)
            measure_ndata += 1

        self.data[trial] += measure_data / measure_ndata / runs

    def _make_learner(self, learner_type):
        if learner_type == 'current_stimulus':
            individual = CurrentStimulusLearner(self.learner_parm)
        elif learner_type == 'trace':
            individual = TraceLearner(self.learner_parm)
        elif learner_type == 'time_tagging':
            individual = TimeTaggingLearner(self.learner_parm)
        elif learner_type == 'flexible_sequence':
            individual = FlexibleSequenceLearnerOneTag(self.learner_parm)
        elif learner_type == 'unique_sequence':
            individual = UniqueSequenceLearner(self.learner_parm)
        else:
            raise Exception(f"Invalid value '{learner_type}' for parameter 'learner_type'.")
        return individual

    def simulate(self):
        n_runs = self.sim_parm.get('runs')
        n_trials = self.sim_parm.get('trials')
        learner_type = self.learner_parm.get('learner_type')

        progress_print("")

        for run_no in range(n_runs):
            individual = self._make_learner(learner_type)

            if run_no == 0:
                self.individual = individual

            self.collect_data(0, individual)

            for trial in range(1, n_trials + 1):
                perc_run = run_no / n_runs
                perc_trial = (trial - 1) / n_trials
                perc = (perc_run + perc_trial / n_runs) * 100
                if perc >= 99:
                    perc = 100
                depth_str = f"{self.learner_parm.get('depth')}"
                progress_print(f"{self.learner_parm.get('learner_type')}(depth={depth_str}): {round(perc, 1)}% \r", end="")

                seq, is_rewarded = self.world.get_next_sequence()
                individual.respond_to_sequence(seq, is_rewarded)

                self.collect_data(trial, individual)

        if self.sim_parm.get('collect_when') != 'last':

            # Remove None's in self.data
            self.data_x = list()
            self.data_y = list()
            for i, y in enumerate(self.data):
                if y is not None:
                    self.data_x.append(i)
                    self.data_y.append(y)

            if self.sim_parm.get('make_sliding'):
                # Make collected data sliding average
                self._make_collected_data_sliding()

    def _make_collected_data_sliding(self):
        progress_print("")
        progress_print("Making collected data sliding.")
        make_sliding = self.sim_parm.get('make_sliding')
        if make_sliding is True:
            sample_size = len(self.data_y)
        else:
            sample_size = make_sliding
        self.data_y = sliding_average(self.data_y, sample_size)


class Parameters():
    def __init__(self, DEFAULTS, **kwargs):
        self.parm = dict(DEFAULTS)
        self.set(**kwargs)

    def set(self, **kwargs):
        for name, value in kwargs.items():
            self._assert(name)
            self.parm[name] = value

    def get(self, name):
        self._assert(name)
        return self.parm[name]

    def _assert(self, name):
        assert(name in self.parm), f"Invalid parameter name '{name}'."

    def get_info(self):
        return str(self.parm)


class WorldParameters(Parameters):
    DEFAULTS = {
        # Type: 'sequence' or 'vector'
        'world_type': 'sequence',

        # The number of additional informative sequences for each depth, e.g. [32, 8, 2, 0, 0]
        'n_inf_sequences': None,

        # Number of noise stimuli to use in the sequences
        'n_noise_stimuli': 100,

        # Number of test sequences. Use 'all' for all sequences in the world.
        'n_test_sequences': 'all',

        # Whether or not to reuse informative stimuli on each level. Only applicable when world_type='vector'.
        'reuse_informative_stimuli': False,

        # Number of informative stimuli. Only applicable when world_type='vector'.
        'n_inf_stimuli': 'free',

        # Percentage stimuli from "sequence world". Only applicable when world_type='parameter_study_mixed'.
        'p': 1,
    }

    def __init__(self, **kwargs):
        super().__init__(WorldParameters.DEFAULTS, **kwargs)


class SimulationParameters(Parameters):
    DEFAULTS = {
        # Number of individuals to simulate
        'runs': 10,

        # Number of trials for each individual
        'trials': 1000,

        # When to collect data: 'each' (each trial), 'last' (only last trial),
        # or integer n (at every nth trial)
        'collect_when': 'every100',

        # Make sliding average of collected data with this sample size. Use False to avoid
        # sliding average.
        'make_sliding': False,

        # Number of exposures to noise sequences during simulation of the world 'parameter_study_mixed'
        'n_culture_noise_exposures': 128
    }

    def __init__(self, **kwargs):
        super().__init__(SimulationParameters.DEFAULTS, **kwargs)


class LearnerParameters(Parameters):
    DEFAULTS = {
        # The learner type. 'current_stimulus_learner', 'trace_learner', 'time_tagging_learner',
        # or 'time_tagging_learner_trace'
        'learner_type': 'current_stimulus_learner',

        # Learning rate
        'alpha': 0.2,

        # Exploration factor
        'beta': 0.75,

        # The u-value for the stimulus that comes after the behavior "no response"
        'value_no_response': 1,

        # The u-value for the stimulus that comes after the behavior "response" if rewarded
        'value_rewarded_response': 1,

        # The u-value for the stimulus that comes after the behavior "response" if not rewarded
        'value_punished_response': -1,

        # Initial stimulus-response value for the behavior 'response' to each stimulus.
        'start_sr_response': 0,

        # Initial stimulus-response value for the behavior 'no response' to each stimulus.
        'start_sr_noresponse': 0,

        # The trace value, used when learner is trace_learner or time_tagging_learner_trace
        'trace': 0.5,

        # The number of time steps to consider for learner_type=symbolic_sequence_learner,
        # time_tagging_learner, time_tagging_learner_groups, and time_tagging_learner_trace
        'depth': 4
    }

    def __init__(self, **kwargs):
        super().__init__(LearnerParameters.DEFAULTS, **kwargs)


# ------------------------------------------------------------
#                          Plots
# ------------------------------------------------------------
def simulate_world(learner_parm, sim_parm, world):
    simulation = Simulation(world, sim_parm, learner_parm)
    simulation.simulate()
    if sim_parm.get('collect_when') == 'last':
        return simulation.data
    else:
        return simulation.data_x, simulation.data_y


def fig_nature(learner_parm, sim_parm):
    vector = [2, 4, 8, 16]
    world_parm = WorldParameters(world_type='vector',
                                 reuse_informative_stimuli=True,
                                 n_inf_sequences=vector,
                                 n_noise_stimuli=50,
                                 n_test_sequences=5 * sum(vector))  # 'all' if no noise stimili

    world = World(world_parm)
    f = plt.figure(figsize=(10, 9))
    learner_parm.set(learner_type='current_stimulus')
    data_x, data_y = simulate_world(learner_parm, sim_parm, world)
    plt.plot(data_x, data_y, label='Depth-1')

    learner_parm.set(learner_type='trace')
    data_x, data_y = simulate_world(learner_parm, sim_parm, world)
    plt.plot(data_x, data_y, label='Trace')

    learner_parm.set(learner_type='time_tagging')
    for depth in [2, 3, 4]:
        label = f"Depth-{depth}"
        learner_parm.set(depth=depth)
        data_x, data_y = simulate_world(learner_parm, sim_parm, world)
        plt.plot(data_x, data_y, label=label)
    plt.title(f"Information distribution {world_parm.get('n_inf_sequences')}")
    plt.xlabel("Trial")
    plt.ylabel("Proportion correct responses")
    plt.ylim([0.45, 1.05])
    plt.legend()
    plt.grid()

    f.tight_layout()
    plt.savefig('Learning-in-nature.pdf')

# ==========================================================================================================


class SimulationParameterStudyMixed(Simulation):
    def __init__(self, world, sim_parm, learner_parm):
        super().__init__(world, sim_parm, learner_parm)

    def simulate(self):
        n_runs = self.sim_parm.get('runs')
        n_trials = self.sim_parm.get('trials')
        learner_type = self.learner_parm.get('learner_type')
        n_culture_noise_exposures = self.sim_parm.get('n_culture_noise_exposures')

        progress_print("")

        for run_no in range(n_runs):
            individual = self._make_learner(learner_type)

            if run_no == 0:
                self.individual = individual

            self.collect_data(0, individual)
            for trial in range(1, n_trials + 1):
                perc_run = run_no / n_runs
                perc_trial = (trial - 1) / n_trials
                perc = (perc_run + perc_trial / n_runs) * 100
                if perc >= 99:
                    perc = 100
                depth_str = f"{self.learner_parm.get('depth')}"
                progress_print(f"{self.learner_parm.get('learner_type')}(depth={depth_str}): {round(perc, 1)}% \r", end="")

                seq, is_rewarded, is_culture_sequence = self.world.get_next_sequence_culture_p()
                if is_culture_sequence:
                    for i in range(n_culture_noise_exposures):
                        seq_noise, is_rewarded_noise = self.world.get_next_culture_sequence_noise()
                        individual.respond_to_sequence(seq_noise, is_rewarded_noise)

                individual.respond_to_sequence(seq, is_rewarded)

                self.collect_data(trial, individual)

        if self.sim_parm.get('collect_when') != 'last':

            # Remove None's in self.data
            self.data_x = list()
            self.data_y = list()
            for i, y in enumerate(self.data):
                if y is not None:
                    self.data_x.append(i)
                    self.data_y.append(y)

            if self.sim_parm.get('make_sliding'):
                # Make collected data sliding average
                self._make_collected_data_sliding()


def fig_culture(learner_parm, sim_parm):

    def _simulate_world(learner_parm, sim_parm, world):
        simulation = SimulationParameterStudyMixed(world, sim_parm, learner_parm)
        simulation.simulate()
        if sim_parm.get('collect_when') == 'last':
            return simulation.data
        else:
            return simulation.data_x, simulation.data_y

    f = plt.figure(figsize=(22, 9))

    learner_types = [('trace', 1), ('unique_sequence', 4), ('flexible_sequence', 4)]
    labels = ['Trace', 'Unique-4', 'Flexible sequence']

    p_values = [0, 0.25, 0.5, 0.75, 1]
    y_values = {learner_type: [] for learner_type in learner_types}

    cnt = 0
    for p in p_values:
        world_parm = WorldParameters(world_type='parameter_study_mixed',
                                     p=p,
                                     n_noise_stimuli=100,
                                     n_test_sequences=24 * 4)
        world = World(world_parm)

        plt.subplot(1, len(p_values), cnt + 1)
        cnt += 1
        for learner_type, label in zip(learner_types, labels):
            learner_parm.set(learner_type=learner_type[0])
            learner_parm.set(depth=learner_type[1])

            x_data, y_data = _simulate_world(learner_parm, sim_parm, world)
            y_values[learner_type].append(y_data[-1])
            plt.plot(x_data, y_data, label=label)
        if cnt == 1:
            plt.legend()
            plt.ylabel("Proportion correct responses")
        plt.ylim([0.45, 1.05])
        plt.grid()
        plt.xlabel('Trial')
        plt.title(f"p={p}")

        f.tight_layout()
        plt.savefig('Parameter_mixed.pdf')


def fig_nature_parameter_study(learner_parm, sim_parm):
    vectors = [[0, 0, 0, 32],
               [8, 8, 8, 8],
               [32, 0, 0, 0]]

    learner_types = [('trace', 1), ('current_stimulus', 1),
                     ('unique_sequence', 2), ('unique_sequence', 3), ('unique_sequence', 4)]
    labels = ['Trace', 'Depth-1', 'Unique-2', 'Unique-3', 'Unique-4']

    trials = 20_000
    sim_parm.set(trials=trials)

    f = plt.figure(figsize=(22, 9))
    plt.title(f"{sim_parm.get('trials')} trials")

    y_values = {learner_type: [] for learner_type in learner_types}
    cnt = 0
    for vector in vectors:
        # reuse_informative_stimuli = (vector == 8)
        world_parm = WorldParameters(world_type='vector',
                                     reuse_informative_stimuli=False,
                                     n_inf_sequences=vector,
                                     n_inf_stimuli=16,
                                     n_noise_stimuli=50,
                                     n_test_sequences=5 * sum(vector))  # 'all' if no noise stimili
        world = World(world_parm)

        plt.subplot(1, 3, cnt + 1)
        cnt += 1
        for learner_type, label in zip(learner_types, labels):
            learner_parm.set(learner_type=learner_type[0])
            learner_parm.set(depth=learner_type[1])
            x_data, y_data = simulate_world(learner_parm, sim_parm, world)
            y_values[learner_type].append(y_data[-1])
            plt.plot(x_data, y_data, label=label)
        if cnt == 1:
            plt.legend()
            plt.ylabel("Proportion correct responses")
        plt.ylim([0.45, 1.05])
        plt.grid()
        plt.xlabel('Trial')
        plt.title(f"Information distribution {vector}")

    f.tight_layout()
    plt.savefig('Learning-in-nature_parameter.pdf')


def cumsum(x_list):
    out = []
    cnt = 0
    for x in x_list:
        cnt += x
        out.append(cnt)
    return out


def fig_revise(learner_parm, sim_parm):
    vectors = [[2, 4, 8, 16], [2, 4, 8, 16]]
    n_noise_stimulis = [18, 498]
    trialss = [10_000, 100_000]

    learner_types = [('unique_sequence', 1), ('unique_sequence', 2), ('unique_sequence', 3), ('unique_sequence', 4)]
    labels = ['Depth-1', 'Unique-2', 'Unique-3', 'Unique-4']

    sim_parm.set(runs=20)
    learner_parm.set(alpha=0.1)

    f = plt.figure(figsize=(16, 8))
    plt.title(f"{sim_parm.get('trials')} trials")

    y_values = {learner_type: [] for learner_type in learner_types}
    cnt = 0
    for n_noise_stimuli, trials, vector in zip(n_noise_stimulis, trialss, vectors):
        sim_parm.set(trials=trials)
        world_parm = WorldParameters(world_type='vector',
                                     reuse_informative_stimuli=True,
                                     n_inf_sequences=vector,
                                     n_inf_stimuli=2,  # XXX 16
                                     n_noise_stimuli=n_noise_stimuli,
                                     n_test_sequences=5 * sum(vector))  # 'all' if no noise stimili
        world = World(world_parm)

        plt.subplot(1, 2, cnt + 1)
        cnt += 1
        for learner_type, label in zip(learner_types, labels):
            learner_parm.set(learner_type=learner_type[0])
            learner_parm.set(depth=learner_type[1])
            x_data, y_data = simulate_world(learner_parm, sim_parm, world)
            # y_data = cumsum(y_data)  # To make U
            y_values[learner_type].append(y_data[-1])
            plt.plot(x_data, y_data, label=label)
        if cnt == 1:
            plt.legend()
            plt.ylabel("Proportion correct responses")
        # plt.plot([0, trials], [0.75, 0.75], linestyle='dashed', color='gray')
        plt.ylim([0.45, 1.05])
        plt.grid()
        plt.xlabel('Trial')
        plt.title(f"{vector}, {n_noise_stimuli + 2} stimuli (2 informative)")

    f.tight_layout()
    plt.savefig('Revised2.pdf')


def N(l, n):
    return n**l


def f(l, r):
    # return 1 - (1 - r)**l
    return 1 - r**l


def U(l, T, n, r, tau):
    tauN = tau * N(l, n)
    return f(l, r) * (T - tauN * (1 - (1 - 1 / tauN)**T))


def fig_panel_in_analytical_model():
    fig = plt.figure(figsize=(20, 9))
    l_values = [1, 2, 3, 4, 5, 6]

    # Plot of U(l, T)
    ax1 = plt.subplot(1, 2, 1)
    T_values = [1000, 2000, 3000, 4000, 5000]
    n = 12
    r = 0.5
    tau = 10
    for T in T_values:
        plt.plot(l_values, [U(l, T, n=n, r=r, tau=tau) for l in l_values],
                 label=f"$T={T}$",
                 marker='o',
                 markevery=1)
    plt.legend()
    plt.title(f"$U(l,T\\,)$ for $n={n}$, $r={r}$, $\\tau={tau}$")
    plt.xlabel("$l$")
    plt.ylabel("$U(l,T\\,)$")

    ax1.text(0, 3300, "(a)", fontsize=25, fontweight='bold', va='top', ha='left')

    # Phase diagram
    ax2 = plt.subplot(1, 2, 2)
    phase_diagram_filled(ax2)

    ax2.text(-1300, 31, "(b)", fontsize=25, fontweight='bold', va='top', ha='left')

    # Adjust spacing between subplots
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.15, hspace=0.1)

    plt.savefig('analytical.pdf')


def _optimal_l(n_values, T_values, l_values, r, tau):
    z = list()
    for n in n_values:
        for T in T_values:
            U_values = [U(l, T, n, r, tau) for l in l_values]

            # Find index (opt_ind) of first local maximum in y
            opt_ind = 0
            last_U = U_values[0]
            for i in range(1, len(U_values)):
                U_value = U_values[i]
                if U_value <= last_U:
                    break
                else:
                    last_U = U_value
                    opt_ind = i

            l_opt = l_values[opt_ind]
            z.append(l_opt)

    z = np.array(z)
    maxz = max(z)
    Z = z.reshape(len(T_values), len(n_values))
    TT, nn = np.meshgrid(T_values, n_values)
    return TT, nn, Z, maxz


def phase_diagram_filled(ax=None):
    tau = 10
    r = 0.5

    l_values = list(range(1, 9))
    T_min, T_max = (3, 10000)
    n_min, n_max = (2, 50)
    T_values = np.linspace(T_min, T_max, 500)
    n_values = np.linspace(n_min, n_max, 500)

    TT, nn, Z, maxz = _optimal_l(n_values, T_values, l_values, r, tau)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    levels = range(0, int(maxz) + 1)
    c = ax.contourf(TT, nn, Z, levels=levels, cmap=plt.get_cmap(name='jet', lut=1024))
    # plt.xscale('log')
    cbarticks = [x + 0.5 for x in levels]
    cbar = plt.colorbar(c, ticks=cbarticks, ticklocation=None)
    cbarticklabels = [str(x + 1) for x in levels]
    cbar.set_ticklabels(cbarticklabels)  # vertically oriented colorbar
    cbar.ax.tick_params(bottom=False, top=False, left=False, right=False, which='both')
    cbar.ax.invert_yaxis()
    plt.title(f"Optimal sequence length, $r={r}$, $\\tau={tau}$")
    plt.xlabel("$T$")
    plt.ylabel("$n$")


def fig_tau_r_supplementary():
    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    phase_diagram_tau(ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    phase_diagram_r(ax2)
    plt.savefig('tau_r.pdf')


def phase_diagram_tau(ax=None):
    r = 0.5
    l_values = list(range(1, 10))
    T_min, T_max = (3, 10000)
    n_min, n_max = (2, 50)
    T_values = np.linspace(T_min, T_max, 500)
    n_values = np.linspace(n_min, n_max, 500)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    taus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    taus_to_label = [1, 5, 10]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default cycle colors
    for i, tau in enumerate(taus):
        TT, nn, Z, _ = _optimal_l(n_values, T_values, l_values, r, tau)
        levels = [1.5]  # Between 1 and 2
        c = ax.contour(TT, nn, Z, levels=levels, colors=colors[i % len(colors)])

        lbl = f"$\\tau={tau}$"
        if tau in taus_to_label:
            # Print label along the line
            fmt = {lev: lbl for lev in levels}
            ax.clabel(c, levels, inline=True, fmt=fmt, fontsize=10)

        # Set label for legend
        c.collections[0].set_label(lbl)
    plt.legend(loc='lower right')
    plt.title(f"Optimal sequence length is 1 above curve and >1 below it, $r={r}$")
    plt.xlabel("$T$")
    plt.ylabel("$n$")


def phase_diagram_r(ax=None):
    tau = 10
    l_values = list(range(1, 10))
    T_min, T_max = (3, 10000)
    n_min, n_max = (2, 50)
    T_values = np.linspace(T_min, T_max, 500)
    n_values = np.linspace(n_min, n_max, 500)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rs_to_label = [0.1, 0.5, 0.9]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default cycle colors
    for i, r in enumerate(rs):
        TT, nn, Z, _ = _optimal_l(n_values, T_values, l_values, r, tau)
        levels = [1.5]  # Between 1 and 2
        c = ax.contour(TT, nn, Z, levels=levels, colors=colors[i % len(colors)])

        lbl = f"$r={r}$"
        if r in rs_to_label:
            # Print label along the line
            fmt = {lev: lbl for lev in levels}
            ax.clabel(c, levels, inline=True, fmt=fmt, fontsize=10)

        # Set label for legend
        c.collections[0].set_label(lbl)
    plt.legend(loc='upper left')
    plt.title(f"Optimal sequence length is 1 above curve and >1 below it, $\\tau={tau}$")
    plt.xlabel("$T$")
    plt.ylabel("$n$")


def solveABCD():
    # A = 1/2(A+B+C+D)
    # A+B = 3/4(A+B+C+D)
    # A+B+C = 7/8(A+B+C+D)
    D = 5

    S3 = 7 * D
    S = S3 + D
    S2 = 3 / 4 * S
    C = S3 - S2
    A = S / 2
    B = S2 - A
    return A, B, C, D


def intersection12(learner_parm, sim_parm):

    def find_intersection(x, y1, y2):
        for x_value, y_value1, y_value2 in zip(x, y1, y2):
            if y_value2 > y_value1:
                return x_value
        return None

    vector = [2, 4, 8, 16]

    sim_parm.set(runs=5, make_sliding=False)
    learner_parm.set(alpha=0.1,
                     learner_type='unique_sequence')

    # ns = [10, 50, 100, 200, 500, 700, 1000, 2000]
    # maxTs = [800_000] * len(ns)  # 25_000, 50_000, 125_000, 175_000, 250_000]

    # ns = [2000]
    # maxTs = [1_000_000]  # 25_000, 50_000, 125_000, 175_000, 250_000]

    # known_u = {10: 1_800, 50: 7_250, 100: 16_250,
    #          200: 31_000, 500: 88_000, 700: 130_500, 1000: 166_000}

    known_U = {10: 4350, 50: 20_400, 100: 38_000, 200: 78_000, 500: 232_500, 700: 280_000, 1000: 440_000, 2000: 770_000}
    # known_u_16 = {10: 9689, 50: 48_000, 100: 106_135, 200: 208_300, 500: 518_000}

    ns = [10, 50, 100, 200, 500, 1000]
    maxTs = [70_000] * len(ns)
    known_u_noslid = {10: 650, 50: 3200, 100: 6500, 200: 12200, 500: 29500, 1000: 66000}

    T_intersection = []
    for n, T in zip(ns, maxTs):
        if n in known_u_noslid:
            T_int = known_u_noslid[n]
        
        else:
            sim_parm.set(trials=T)
            world_parm = WorldParameters(world_type='vector',
                                        reuse_informative_stimuli=True,
                                        n_inf_sequences=vector,
                                        n_inf_stimuli=2,
                                        n_noise_stimuli=n,
                                        n_test_sequences=5 * sum(vector))  # 'all' if no noise stimili
            world = World(world_parm)

            learner_parm.set(depth=1)
            x_data1, y_data1 = simulate_world(learner_parm, sim_parm, world)
            # y_data1 = cumsum(y_data1)  # To make U

            learner_parm.set(depth=2)
            x_data2, y_data2 = simulate_world(learner_parm, sim_parm, world)
            # y_data2 = cumsum(y_data2)  # To make U

            assert(x_data1 == x_data2)

            plt.figure()
            plt.plot(x_data1, y_data1)
            plt.plot(x_data2, y_data2)
            plt.grid()

            T_int = find_intersection(x_data1, y_data1, y_data2)
        
        T_intersection.append(T_int)        

    plt.figure()
    plt.plot(ns, T_intersection, marker='o')
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("T when unique(2) intersects current")
    # plt.title("Stora U")




# ------------------------------------------------------------
if __name__ == '__main__':
    # A, B, C, D = solveABCD()
    # print(f"{(D, C, B, A)}")

    # fig_panel_in_analytical_model()
    # fig_tau_r_supplementary()

    learner_parm = LearnerParameters(value_rewarded_response=5,
                                     value_punished_response=-4,
                                     value_no_response=0,
                                     beta=1,
                                     alpha=0.2,
                                     start_sr_response=-1,
                                     start_sr_noresponse=0,
                                     trace=0.5)
    sim_parm = SimulationParameters(runs=100,
                                    trials=20_000,
                                    make_sliding=True)

    # XXX
    # intersection12(learner_parm, sim_parm)

    # ---------- Nature-simulation ----------
    # fig_nature(learner_parm, sim_parm)
    fig_nature_parameter_study(learner_parm, sim_parm)
    # fig_revise(learner_parm, sim_parm)

    # ---------- Culture-simulation ----------
    # sim_parm.set(trials=60_000,
    #              n_culture_noise_exposures=10)
    # fig_culture(learner_parm, sim_parm)

    plt.show()
