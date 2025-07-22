# ------------------------------------------------------------------------
# Script to evaluate the results leakage experiments stored in neptune.
# ------------------------------------------------------------------------
# Authored by: Marius Bock and Maximilian Hopp
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import neptune
import argparse

import numpy as np


class lnAccSubjects():
    def __init__(self, lnAcc_val, name, single_subject_values):
        self.lnAcc = lnAcc_val
        self.attack = name
        self.single_subject_values = single_subject_values

class Evaluation():
    def __init__(self):
        pass

    def calculate_accuracy_metrics(self, parser_args):
        # Load old run 
        self.parser = parser_args
        run_id = parser_args.run_id
        
        old_run = neptune.init_run(
            project=parser_args.project,
            api_token=parser_args.api_token,
            with_id=run_id, 
            mode="read-only"
        )

        # Access parameters, metadata, or logged data
        self.attacks = old_run["label_attack"].fetch()
        
        # Print parameters
        self.args = old_run["args"].fetch()
        print('Params:', self.args)

        print("ClassAvgAcc")
        # Calculate label number accuracy over all subjects for each attack
        self.classAvgAcc = 0
        sbjs_number = 0
        for attack in parser_args.label_strat_array:
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    self.classAvgAcc += self.attacks[attack][sbjs]["final_classAvgAcc"]
                    sbjs_number += 1
            # print as percentage in format xx.xx%
            #print('{:.2f}%'.format((self.classAvgAcc / sbjs_number) * 100))
            single_subject_values = []
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    single_subject_values.append(self.attacks[attack][sbjs]["final_classAvgAcc"])
            print('{:.2f}%'.format(np.std(single_subject_values) * 100))
            sbjs_number = 0
            self.classAvgAcc = 0
        
        print("LnAcc")   
        # Calculate label number accuracy over all subjects for each attack
        self.ln_all = 0
        sbjs_number = 0
        for attack in parser_args.label_strat_array:
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    self.ln_all += self.attacks[attack][sbjs]["final_lnAcc"]
                    sbjs_number += 1
            # print as percentage in format xx.xx%
            #print('{:.2f}%'.format((self.ln_all / sbjs_number) * 100))
            single_subject_values = []
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    single_subject_values.append(self.attacks[attack][sbjs]["final_lnAcc"])
            print('{:.2f}%'.format(np.std(single_subject_values) * 100))
            sbjs_number = 0
            self.ln_all = 0

        print("LeAcc")   
        # Calculate label number accuracy over all subjects for each attack
        self.le_all = 0
        sbjs_number = 0
        for attack in parser_args.label_strat_array:
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    self.le_all += self.attacks[attack][sbjs]["final_leAcc"]
                    sbjs_number += 1
            # print as percentage in format xx.xx%
            #print('{:.2f}%'.format((self.le_all / sbjs_number) * 100))
            single_subject_values = []
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    single_subject_values.append(self.attacks[attack][sbjs]["final_leAcc"])
            print('{:.2f}%'.format(np.std(single_subject_values) * 100))
            sbjs_number = 0
            self.le_all = 0

        """# Calculate label number accuracy over all subjects for each attack
        self.ln_all = 0
        self.le_all = 0
        self.classAvgAcc = 0
        sbjs_number = 0
        for attack in parser_args.label_strat_array:
            for sbjs in self.attacks[attack]:
                if "loso_sbj_" in sbjs:
                    self.ln_all += self.attacks[attack][sbjs]["final_lnAcc"]
                    self.le_all += self.attacks[attack][sbjs]["final_leAcc"]
                    self.classAvgAcc += self.attacks[attack][sbjs]["final_classAvgAcc"]
                    sbjs_number += 1
            print('Attack: {}'.format(attack))
            print('LnAcc: {}'.format(self.ln_all / sbjs_number))
            print('LeAcc: {}'.format(self.le_all / sbjs_number))
            print('ClassAvgAcc: {}'.format(self.classAvgAcc / sbjs_number))
            sbjs_number = 0
            self.ln_all = 0 
            self.le_all = 0
            self.classAvgAcc = 0"""
        old_run.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_strat_array', nargs='+', default=['wainakh-simple', 'wainakh-whitebox', 'ebi', 'iLRG', 'llbgAVG'], type=str, help='List of label strategies to evaluate')
    parser.add_argument('--runs', nargs='+', default=['LAB-156'], type=str, help='List of run IDs to evaluate')
    parser.add_argument('--project', default='wasedo/label-leakage', type=str, help='Project name in Neptune')
    parser.add_argument('--api_token', default='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNzYwZjBiMi0xMzEwLTQyMGEtOTFkMC01M2JjMGQzNzc0OTUifQ==', type=str, help='API token for Neptune')
    args = parser.parse_args()
    
    # Sampling strategy
    for run in args.runs:
        eval = Evaluation()
        args.run_id = run
        eval.calculate_accuracy_metrics(args)
    