import pandas

med_course_titles = [
    "Medicine Overall",
    "Medicine Overall",
    "Medicine Overall",
    "Medicine/MedStats/Summer2014",
    "Medicine/MedStats/Summer2014",
    "Medicine/MedStats/Summer2014",
    "Medicine/HRP258/Statistics_in_Medicine",
    "Medicine/HRP258/Statistics_in_Medicine",
    "Medicine/HRP258/Statistics_in_Medicine",
    "Medicine/SURG210/Managing_Emergencies_What_Every_Doctor_Must_Know",
    "Medicine/SURG210/Managing_Emergencies_What_Every_Doctor_Must_Know",
    "Medicine/SURG210/Managing_Emergencies_What_Every_Doctor_Must_Know",
    "Medicine/SciWrite/Fall2013",
    "Medicine/SciWrite/Fall2013",
    "Medicine/SciWrite/Fall2013"
]

hum_course_titles = [
    "Humanities Overall",
    "Humanities Overall",
    "Humanities Overall",
    "HumanitiesSciences/Econ-1/Summer2014",
    "HumanitiesSciences/Econ-1/Summer2014",
    "HumanitiesSciences/Econ-1/Summer2014",
    "HumanitiesSciences/EP101/Environmental_Physiology",
    "HumanitiesSciences/EP101/Environmental_Physiology",
    "HumanitiesSciences/EP101/Environmental_Physiology",
#    "GlobalHealth/WomensHealth/Winter2014",
#    "GlobalHealth/WomensHealth/Winter2014",
#    "GlobalHealth/WomensHealth/Winter2014",
    "HumanitiesScience/StatLearning/Winter2014",
    "HumanitiesScience/StatLearning/Winter2014",
    "HumanitiesScience/StatLearning/Winter2014",
    "HumanitiesSciences/Econ1V/Summer2014",
    "HumanitiesSciences/Econ1V/Summer2014",
    "HumanitiesSciences/Econ1V/Summer2014",
    "HumanitiesScience/Stats216/Winter2014",
    "HumanitiesScience/Stats216/Winter2014",
    "HumanitiesScience/Stats216/Winter2014"
]

medicine_performance = [
0.667, 0.446, 0.509,
0.72, 0.49606299, 0.58741259,
0.62349398, 0.45594714, 0.52671756,
0.5, 0.11111111, 0.18181818,
0.42372881, 0.21052632, 0.28318584 ]

humanities_performance = [
0.689, 0.593, 0.634,
0.81909548, 0.49244713, 0.61509434,
0.43248945, 0.67656766, 0.52767053,
# 0.48175182, 0.39520958, 0.43421053,
0.70752089, 0.66579292, 0.68602296,
0.85714286, 0.75, 0.8,
0.79,  0.75238095, 0.77073171 ]

hum_metrics = ["precision", "recall", "f1"] * 6
med_metrics = ["precision", "recall", "f1"] * 5

HUMANITIES_COURSES = pandas.DataFrame({"Course Title": pandas.Categorical(hum_course_titles),
                  "Performance":  pandas.Series(humanities_performance),
                  "Metric": pandas.Categorical(hum_metrics)})

MEDICINE_COURSES = pandas.DataFrame({"Course Title": pandas.Categorical(med_course_titles),
                  "Performance":  pandas.Series(medicine_performance),
                  "Metric": pandas.Categorical(med_metrics)})
