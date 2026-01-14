"""
Differentiable Logic Layer

Product t-norm based fuzzy logic for rule composition (Paper Section 5.3, Eq. 2).
Rules are parsed from string formulas into callable ASTs.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional

from ruleground.predicates.ontology import Sport, SPORT_FROM_STR
from ruleground.predicates.rules import Rule, RULE_LIBRARY, get_rules_for_sport


class DifferentiableLogic:
    """Product t-norm based differentiable logic.

    AND(a, b) = a * b          (product t-norm)
    OR(a, b)  = a + b - a * b  (product t-conorm)
    NOT(a)    = 1 - a           (standard negation)
    IMPLIES(a, b) = 1 - a * (1 - b)  (material conditional via t-norm)
    """

    @staticmethod
    def AND(a: Tensor, b: Tensor) -> Tensor:
        return a * b

    @staticmethod
    def OR(a: Tensor, b: Tensor) -> Tensor:
        return a + b - a * b

    @staticmethod
    def NOT(a: Tensor) -> Tensor:
        return 1.0 - a

    @staticmethod
    def IMPLIES(a: Tensor, b: Tensor) -> Tensor:
        return 1.0 - a * (1.0 - b)

    @classmethod
    def multi_AND(cls, *tensors: Tensor) -> Tensor:
        result = tensors[0]
        for t in tensors[1:]:
            result = cls.AND(result, t)
        return result

    @classmethod
    def multi_OR(cls, *tensors: Tensor) -> Tensor:
        result = tensors[0]
        for t in tensors[1:]:
            result = cls.OR(result, t)
        return result


class RuleComposer:
    """Composes rules from predicate probabilities using differentiable logic.

    Parses formula strings (e.g. "contact_occurred AND NOT defender_set") into
    callable ASTs and evaluates them against predicate probability dictionaries.
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.logic = DifferentiableLogic()
        self.rules = rules if rules is not None else RULE_LIBRARY
        self._compiled: Dict[str, Callable] = {}
        self._compile_rules()

    def _compile_rules(self) -> None:
        """Compile all rule formulas into callable functions."""
        for rule in self.rules:
            self._compiled[rule.name] = self._parse_formula(rule.formula)

    def _parse_formula(self, formula: str) -> Callable:
        """Parse a formula string into a callable that evaluates predicate dicts."""
        tokens = formula.replace("(", " ( ").replace(")", " ) ").split()

        def parse_expr(pos: int = 0):
            def parse_atom(pos: int):
                if tokens[pos] == "NOT":
                    operand, pos = parse_atom(pos + 1)
                    return ("NOT", operand), pos
                elif tokens[pos] == "(":
                    result, pos = parse_expr(pos + 1)
                    return result, pos + 1  # skip ')'
                else:
                    return ("PRED", tokens[pos]), pos + 1

            left, pos = parse_atom(pos)
            while pos < len(tokens) and tokens[pos] in ("AND", "OR"):
                op = tokens[pos]
                right, pos = parse_atom(pos + 1)
                left = (op, left, right)
            return left, pos

        ast, _ = parse_expr()

        def evaluate(node, preds: Dict[str, Tensor]) -> Tensor:
            if node[0] == "PRED":
                return preds[node[1]]
            elif node[0] == "NOT":
                return self.logic.NOT(evaluate(node[1], preds))
            elif node[0] == "AND":
                return self.logic.AND(
                    evaluate(node[1], preds), evaluate(node[2], preds)
                )
            elif node[0] == "OR":
                return self.logic.OR(
                    evaluate(node[1], preds), evaluate(node[2], preds)
                )
            else:
                raise ValueError(f"Unknown AST node: {node[0]}")

        return lambda preds: evaluate(ast, preds)

    def __call__(self, predicate_probs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute rule scores from predicate probabilities.

        Args:
            predicate_probs: mapping predicate_name -> Tensor [B] of probabilities

        Returns:
            mapping rule_name -> Tensor [B] of rule scores
        """
        results = {}
        for name, func in self._compiled.items():
            try:
                results[name] = func(predicate_probs)
            except KeyError:
                # Skip rules whose predicates aren't in the dict
                # (e.g. basketball rules when processing a soccer clip)
                continue
        return results

    def for_sport(self, sport: Sport | str) -> "RuleComposer":
        """Return a new RuleComposer with only rules for a given sport."""
        rules = get_rules_for_sport(sport)
        return RuleComposer(rules=rules)

    @property
    def rule_names(self) -> List[str]:
        return list(self._compiled.keys())
