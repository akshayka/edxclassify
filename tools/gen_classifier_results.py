import edxclassify.harness as harness
import os
import sys

hum = 'data/gold_sets/humanities_gold_v9'
med = 'data/gold_sets/medicine_gold_v9'
math = 'data/gold_sets/math_gold_v9'
stats_2013 =\
    'data/gold_sets/partitions/single_courses/'\
    'medicine_gold_v8_Medicine_HRP258_Statistics_in_Medicine'
stats_2014 =\
    'data/gold_sets/partitions/single_courses/'\
    'medicine_gold_v8_Medicine_MedStats_Summer2014'
stat_learning =\
    'data/gold_sets/partitions/single_courses/'\
    'humanities_gold_v8_HumanitiesScience_StatLearning_Winter2014'
stat_216 =\
    'data/gold_sets/partitions/single_courses/'\
    'humanities_gold_v8_HumanitiesScience_Stats216_Winter2014'
econ_1 =\
    'data/gold_sets/partitions/single_courses/'\
    'humanities_gold_v8_HumanitiesSciences_Econ-1_Summer2014'
econ_iv =\
    'data/gold_sets/partitions/single_courses/'\
    'humanities_gold_v8_HumanitiesSciences_Econ1V_Summer2014'
ep101 =\
    'data/gold_sets/partitions/single_courses/'\
    'humanities_gold_v8_HumanitiesSciences_EP101_Environmental_Physiology'
global_health =\
    'data/gold_sets/partitions/single_courses/'\
    'humanities_gold_v8_GlobalHealth_WomensHealth_Winter2014'
emerg =\
    'data/gold_sets/partitions/single_courses/'\
     'medicine_gold_v8_Medicine_SURG210_Managing_Emergencies_What_Every_Doctor_Must_Know'
sciwrite =\
    'data/gold_sets/partitions/single_courses/'\
    'medicine_gold_v8_Medicine_SciWrite_Fall2013'
files = [hum, med, math, stats_2013, stats_2014, stat_learning, stat_216,\
            econ_1, econ_iv, ep101, global_health, emerg, sciwrite]

classes = ['question', 'answer', 'opinion', 'confusion', 'sentiment', 'urgency']
cmd = 'logistic -p 0.27'

triplets = [(stats_2013, stats_2014, 'across_stats'),
         (stats_2014, stats_2013, 'reverse_across_stats'),
         (stat_learning, stat_216, 'learning_216'),
         (stat_216, stat_learning, '216_learning'),
         (econ_1, econ_iv, 'econ_1_iv'),
         (econ_iv, econ_1, 'econ_iv_1'),
         (stat_learning, stats_2014, 'learning_stats_2013'),
         (econ_1, stats_2013, 'econ_1_stats_2013'),
         (global_health, stats_2013, 'global_stats_2013'),
         (stats_2013, global_health, 'stats_2013_global'),
         (global_health, sciwrite, 'global_sciwrite'),
         (ep101, stats_2013, 'ep101_stats_2013')]

# Individual courses / course sets
for f in files:
    for c in classes:
        curcmd = cmd
        if c == 'confusion':
            # Only do stacked + binary for confusion
            curcmd = cmd + ' -c -b' 
        harness_args = [f, c] + curcmd.split(' ')
        output_file = 'gen_data/' + os.path.basename(f) + '_' + c
        if os.path.isfile(output_file):
            print 'found results for ' + ' '.join(harness_args) + '; skipping.'
        else:
            print 'executing ' + ' '.join(harness_args) + ' ...'
            stdout_save = sys.stdout
            sys.stdout = open(output_file, "wb")
            harness.main(harness_args)
            sys.stdout.close()
            sys.stdout = stdout_save

# Go across courses
for course1, course2, name in triplets:
    for c in classes:
        curcmd = cmd
        if c == 'confusion':
            curcmd = cmd + ' -c' 
        harness_args = [course1, c] + curcmd.split(' ') + ['-tst_file', course2]
        output_file = 'gen_data/' + name + '_' + c
        if os.path.isfile(output_file):
            print 'found results for ' + ' '.join(harness_args) + '; skipping.'
        else:
            print 'executing ' + ' '.join(harness_args) + ' ...'
            stdout_save = sys.stdout
            sys.stdout = open(output_file, "wb")
            harness.main(harness_args)
            sys.stdout.close()
            sys.stdout = stdout_save
