"""Script is used to optimize the template for a given pdb file."""

from leopard_em.pydantic_models import OptimizeTemplateManager

OPTIMIZE_YAML_PATH = "optimize_template_example_config.yaml"


def main() -> None:
    """Main function to run the optimize template program."""
    otm = OptimizeTemplateManager.from_yaml(OPTIMIZE_YAML_PATH)
    otm.run_optimize_template(output_text_path="results/optimize_template_results.txt")


if __name__ == "__main__":
    main()
