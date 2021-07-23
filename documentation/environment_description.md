# Environment Description

## Initial condition
The environment is seeded with the question (`Q`) and candidate answer (`A`).
Explanation set is empty

## Iteration Step
1. Generate the query:
   - Concatenate the content terms of `Q + A`
   - Filter out those covered by the phrases in the explanation set
    - If the number of non-covered terms is `<=2`, expand the query with the terms of the last chosen explanation
2. Retrieve phrases with the query
3. Rank the pool of candidate phrases by their `Qs` score (See AIR)
4. Let the policy choose among the different candidate phrases to expand the explanation set or to early stop
5. Check stop condition to either iterate or stop
    
    
## Stop Condition
- If remaining terms are `0`, return success with the explanation set
- If the search timed out _or_ if early stop action chosen, return failure with explanation set

## Reward structure
`TODO`

---
## Necessary elements

- Text pre-processor:
    - Case folder
    - Stemmer
    - Stop-word filter
- Query similarity function: See AIR for details
- Query Coverage function (Trivial)

## Atachments
- [QASC](https://arxiv.org/pdf/1910.11473.pdf)
- [AIR](https://arxiv.org/pdf/2005.01218.pdf)
