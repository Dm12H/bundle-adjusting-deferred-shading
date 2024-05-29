import subprocess


def run_multiple():
    dtu_objects = [
        '110_rabbit', '37_scissors', '63_fruits', '83_smurf', '97_cans',
        '106_birds', '69_snowman', '40_block', '65_skull', '105_plush',
        '114_buddha', '55_bunny', '24_redhouse', '122_owl', '118_angel']
    idx = 0
    it_exps = []
    it_upsaples = []
    for its in range(2500, 5000, 500):
        ups_start_point = its - 2000
        upsample_list = list(range(ups_start_point, ups_start_point, 500))
        it_exps.append(ups_start_point)
        it_upsaples.append(upsample_list)
    while idx < len(it_exps):
        num_its = it_exps[idx]
        upsamples = it_upsaples[idx]
        try:
            subprocess.run(
                ["dvc", "exp", "run", "-f",
                 "--downstream", "run-eval",
                 "--name", f"its_{num_its}",
                 "-S", f"run.run_name=114_buddha",
                 "-S", "run.perturbs.pos=15",
                 "-S", "run.perturbs.dir=15",
                 "-S", f"run.subdir_name=its_{num_its}",
                 "-S", "run.run_name=114_buddha",
                 "-S", "run.image_scale=2",
                 "-S", f"run.upsample_iterations={upsamples}",
                 "-S", f"run.rebuild_iterations={[]}",
                 "-S", "run.image_scale=2",
                 "-S", f"run.iterations={num_its}"],
                timeout=1000)
        except subprocess.TimeoutExpired:
            print(f"Had to stop exp its_{num_its}, rerunning")
            continue
        idx += 1
        subprocess.run(
            ["dvc", "exp", "save", "--name", f"its_{num_its}"]
        )


if __name__ == "__main__":
    run_multiple()
