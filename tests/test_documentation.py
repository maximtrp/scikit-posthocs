import ast
import doctest
import inspect
import io
import re

import pytest

import scikit_posthocs as sp


@pytest.mark.parametrize(
    "function",
    [
        sp.posthoc_ttest,
        sp.posthoc_tukey_hsd,
        sp.posthoc_anderson,
        sp.posthoc_vanwaerden,
        sp.test_mackwolfe,
        sp.sign_array,
        sp.sign_table,
        sp.outliers_iqr,
        sp.outliers_grubbs,
    ],
)
def test_documented_examples(function):
    doc = doctest.DocTestParser().get_doctest(function.__doc__, {}, function.__name__, None, 0)
    report = io.StringIO()
    result = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE).run(
        doc, out=report.write
    )
    if result.failed != 0:
        pytest.fail(report.getvalue())


def test_public_documented_parameters_match_signatures():
    for name, function in inspect.getmembers(sp, inspect.isfunction):
        if name.startswith("_"):
            continue
        doc = inspect.getdoc(function) or ""
        if "Parameters\n----------" not in doc:
            continue
        section = doc.split("Parameters\n----------", 1)[1].split("\nReturns\n-------", 1)[0]
        documented = set(re.findall(r"^(\w+)\s*:", section, re.MULTILINE))
        documented.update(set(re.findall(r"^(\w+)\s*$", section, re.MULTILINE)) & {"kwargs"})
        parameters = inspect.signature(function).parameters
        actual = set(parameters)
        if documented != actual:
            pytest.fail(f"{name}: missing {actual - documented}, extra {documented - actual}")
        for parameter, default in re.findall(
            r"^(\w+)\s*:[^\n]*?\s=\s*(True|False|None|\d+(?:\.\d+)?)\s*$",
            section,
            re.MULTILINE,
        ):
            if str(parameters[parameter].default) != default:
                pytest.fail(f"{name}: {parameter} default")


def test_public_examples_have_valid_syntax_and_imports():
    for name, function in inspect.getmembers(sp, inspect.isfunction):
        if name.startswith("_"):
            continue
        doc = inspect.getdoc(function) or ""
        if "Examples\n--------" not in doc:
            continue
        examples = doctest.DocTestParser().get_examples(doc.split("Examples\n--------", 1)[1])
        imports = set()
        for example in examples:
            tree = ast.parse(example.source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in {"np", "pd", "sp", "ph"}
                    and node.value.id not in imports
                ):
                    pytest.fail(f"{name}: missing import for {node.value.id}")
            imports.update(
                alias.asname or alias.name.split(".")[0]
                for node in tree.body
                if isinstance(node, ast.Import)
                for alias in node.names
            )
