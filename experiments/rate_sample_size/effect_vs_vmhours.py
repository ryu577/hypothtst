from hypothtst.tst.stochparams.rate.ump import UMPPoisson
import numpy as np
import matplotlib.pyplot as plt


def effect_hours_plot():
    vm_centuries=np.arange(.08,20,0.8)
    effects = np.array([UMPPoisson.effect_vmhours_equal_t(t=t,lmb_base=1,max_effect=120)\
                for t in vm_centuries])
    blade_days = vm_centuries*365*100/10 ##Assuming 10 VMs per node.
    plt.plot(blade_days,effects)
    plt.xlabel("Blade days (each group)")
    plt.ylabel("AIR change")


