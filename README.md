## bayes_vs_cma

### quick start

* run bayes_opt

```
$ python bayes_opt.py
```

* run cma-es

```
$ python cma-es.py
```

The example output is

```
$ python cma-es.py                             12:14:49  ☁  master ☂ ✭
create directory which is log/2D-HimmelblauFunction_2017-08-11-12-16
6 :  53.9618311152
12 :  53.9618311152
18 :  53.6564062568
24 :  36.1323293002
30 :  14.6497939751
36 :  14.4362912843
42 :  14.4362912843
48 :  3.76755699947
54 :  0.0533554188594
seed is 695037379
create directory which is log/2D-HimmelblauFunction_2017-08-11-12-16
```

If you want to start viewer, please run following command.

```
$ python himmelblau_viewer.py log/2D-HimmelblauFunction_2017-08-11-12-16
```

You should change the path `log/2D-HimmelblauFunction_2017-08-11-12-16` to your own path.