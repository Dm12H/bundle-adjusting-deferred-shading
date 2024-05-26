import subprocess


def run_multiple():
    dtu_objects = [
        '110_rabbit', '37_scissors', '63_fruits', '83_smurf', '97_cans',
        '106_birds', '69_snowman', '40_block', '65_skull', '105_plush',
        '114_buddha', '55_bunny', '24_redhouse', '122_owl', '118_angel']
    idx = 0
    errs = [0.1, 0.2, 0.5] + list(range(1, 16))
    while idx < len(dtu_objects):
        cur_err = errs[idx]
        try:
            subprocess.run(
                ["dvc", "exp", "run", "-f",
                 "--downstream", "run-eval",
                 "--name", f"err_{cur_err:.01f}",
                 "-S", f"run.run_name=114_buddha",
                 "-S", f"run.perturbs.pos=f{cur_err}",
                 "-S", f"run.perturbs.dir=f{cur_err}",
                 "-S", f"run.subdir_name=err_{cur_err:.01f}",
                 "-S", f"run.run_name=114_buddha",
                 "-S", "run.image_scale=2",
                 "-S", "run.iterations=2500"],
                timeout=1000)
        except subprocess.TimeoutExpired:
            print(f"Had to stop exp err_{cur_err:.01f}, rerunning")
            continue
        idx += 1
        subprocess.run(
            ["dvc", "exp", "save", "--name", f"err_{cur_err:.01f}"]
        )


if __name__ == "__main__":
    run_multiple()
