"""Reusable request payload fixtures."""

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_5ac8f214",
#     "problem_id": "binary_search_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "wrong_answer",

#     "source_code": """def binary_search(nums, target):
#     left = 0
#     right = len(nums)

#     while left < right:
#         mid = (left + right) // 2

#         if nums[mid] == target:
#             return mid
#         elif nums[mid] < target:
#             left = mid + 1
#         else:
#             right = mid

#     return -1
# """,

#     "test_summary": {
#         "total_test_cases": 30,
#         "passed_test_cases": 24,
#         "failed_test_cases": 6,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "nums=[1]\ntarget=1",
#             "expected_output": "0",
#             "actual_output": "-1",
#         },
#         {
#             "stdin": "nums=[1,3,5,7]\ntarget=7",
#             "expected_output": "3",
#             "actual_output": "-1",
#         },
#         {
#             "stdin": "nums=[2,4,6,8,10]\ntarget=10",
#             "expected_output": "4",
#             "actual_output": "-1",
#         },
#     ],

#     "stdout": "-1",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 39,
#     "memory_kb": 14208,

#     "submitted_at": "2026-07-02T10:15:43Z",
# }


# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_runtime_001",
#     "problem_id": "two_sum_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "runtime_error",

#     "source_code": """def solve(nums):
#     for i in range(len(nums)):
#         print(nums[i + 1])
# """,

#     "test_summary": {
#         "total_test_cases": 15,
#         "passed_test_cases": 3,
#         "failed_test_cases": 12,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "1 2 3",
#             "expected_output": "1\n2\n3",
#             "actual_output": "",
#         }
#     ],

#     "stdout": "",
#     "stderr": "IndexError: list index out of range",
#     "compile_output": "",

#     "execution_time_ms": 2,
#     "memory_kb": 9120,

#     "submitted_at": "2026-07-04T09:00:00Z",
# }

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_compile_001",
#     "problem_id": "sum_array_001",
#     "problem_statement": (
#         "Given an array of integers nums, return the sum of all the elements "
#         "in the array."
#     ),
#     "user_id": "user_558",
#     "language": "python",
#     "verdict": "compilation_error",
#     "source_code": """def solve(nums)
#     return sum(nums)
# """,
#     "test_summary": {
#         "total_test_cases": 0,
#         "passed_test_cases": 0,
#         "failed_test_cases": 0,
#     },
#     "sample_failed_cases": [],
#     "stdout": "",
#     "stderr": "",
#     "compile_output": "SyntaxError: expected ':'",
#     "execution_time_ms": 0,
#     "memory_kb": 0,
#     "submitted_at": "2026-07-04T09:15:00Z",
# }


# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_tle_001",
#     "problem_id": "factorial_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "time_limit_exceeded",

#     "source_code": """def solve(n):
#     while True:
#         pass
# """,

#     "test_summary": {
#         "total_test_cases": 20,
#         "passed_test_cases": 5,
#         "failed_test_cases": 15,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "100000",
#             "expected_output": "933262154439...",
#             "actual_output": "",
#         }
#     ],

#     "stdout": "",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 2000,
#     "memory_kb": 11000,

#     "submitted_at": "2026-07-04T09:30:00Z",
# }

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_accept_001",
#     "problem_id": "sum_array_001",
#     "user_id": "user_558",

#     "language": "python",
#     "verdict": "accepted",

#     "source_code": """def solve(nums):
#     return sum(nums)
# """,

#     "test_summary": {
#         "total_test_cases": 20,
#         "passed_test_cases": 20,
#         "failed_test_cases": 0,
#     },

#     "sample_failed_cases": [],

#     "stdout": "15",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 8,
#     "memory_kb": 10240,

#     "submitted_at": "2026-07-04T09:45:00Z",
# }

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_runtime_002",
#     "problem_id": "sum_array",
#     "user_id": "user_222",

#     "language": "python",
#     "verdict": "runtime_error",

#     "source_code": """def solve(nums):
#     return summ(nums)
# """,

#     "test_summary": {
#         "total_test_cases": 12,
#         "passed_test_cases": 0,
#         "failed_test_cases": 12,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "[1,2,3]",
#             "expected_output": "6",
#             "actual_output": "",
#         }
#     ],

#     "stdout": "",
#     "stderr": """NameError: name 'summ' is not defined""",
#     "compile_output": "",

#     "execution_time_ms": 2,
#     "memory_kb": 10240,

#     "submitted_at": "2026-07-04T15:40:00Z",
# }

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_median_001",
#     "problem_id": "median_sorted_arrays",
#     "problem_statement": (
#         "Given two sorted arrays nums1 and nums2, return the median of the two "
#         "sorted arrays. The overall run time complexity should be O(log(m+n)). "
#         "The median is the middle value in the merged sorted order, or the "
#         "average of the two middle values when the total length is even."
#     ),
#     "user_id": "user_314",

#     "language": "cpp",
#     "verdict": "wrong_answer",

#     "source_code": """class Solution {
# public:
#     int p1 = 0, p2 = 0;

#     int getMin(vector<int>& nums1, vector<int>& nums2) {
#         if (p1 < nums1.size() && p2 < nums2.size()) {
#             return nums1[p1] <= nums2[p2] ? nums1[p1++] : nums2[p2++];
#         } else if (p1 < nums1.size()) {
#             return nums1[p1++];
#         } else if (p2 < nums2.size()) {
#             return nums2[p2++];
#         }
#         return -1;
#     }

#     double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
#         int m = nums1.size(), n = nums2.size();

#         if ((m + n) % 2 == 0) {

#             for (int i = 0; i < (m + n) / 2; i++) {
#                 getMin(nums1, nums2);
#             }

#             return (getMin(nums1, nums2) + getMin(nums1, nums2)) / 2.0;

#         } else {

#             for (int i = 0; i < (m + n) / 2; i++) {
#                 getMin(nums1, nums2);
#             }

#             return getMin(nums1, nums2);
#         }
#     }
# };
# """,

#     "test_summary": {
#         "total_test_cases": 40,
#         "passed_test_cases": 35,
#         "failed_test_cases": 5,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "nums1 = [1,3]\nnums2 = [2,4]",
#             "expected_output": "2.5",
#             "actual_output": "3.5",
#         },
#         {
#             "stdin": "nums1 = [1,2]\nnums2 = [3,4]",
#             "expected_output": "2.5",
#             "actual_output": "3.5",
#         },
#         {
#             "stdin": "nums1 = [1,2,3]\nnums2 = [4,5,6]",
#             "expected_output": "3.5",
#             "actual_output": "4.5",
#         },
#     ],

#     "stdout": "3.5",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 8,
#     "memory_kb": 12672,

#     "submitted_at": "2026-07-04T16:30:00Z",
# }


# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_sliding_window_001",
#     "problem_id": "longest_substring_without_repeating",
#     "problem_statement": (
#         "Given a string s, find the length of the longest substring without "
#         "duplicate characters."
#     ),
#     "user_id": "user_901",

#     "language": "cpp",
#     "verdict": "wrong_answer",

#     "source_code": """class Solution {
# public:
#     int lengthOfLongestSubstring(string s) {
#         int n = s.length();
#         int maxLength = 0;
#         unordered_map<char, int> charMap;
#         int left = 0;

#         for (int right = 0; right < n; right++) {

#             if (charMap.count(s[right]) == 0) {
#                 charMap[s[right]] = right;
#                 maxLength = max(maxLength, right - left + 1);
#             } else {
#                 left = charMap[s[right]] + 1;
#                 charMap[s[right]] = right;
#                 maxLength = max(maxLength, right - left + 1);
#             }
#         }

#         return maxLength;
#     }
# };
# """,

#     "test_summary": {
#         "total_test_cases": 45,
#         "passed_test_cases": 39,
#         "failed_test_cases": 6,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": 's = "abba"',
#             "expected_output": "2",
#             "actual_output": "3",
#         },
#         {
#             "stdin": 's = "tmmzuxt"',
#             "expected_output": "5",
#             "actual_output": "6",
#         },
#         {
#             "stdin": 's = "dvdf"',
#             "expected_output": "3",
#             "actual_output": "4",
#         },
#     ],

#     "stdout": "4",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 7,
#     "memory_kb": 11840,

#     "submitted_at": "2026-07-05T10:45:00Z",
# }

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_roman_001",
#     "problem_id": "integer_to_roman",
#     "problem_statement": (
#         "Given an integer, convert it to a Roman numeral. Roman numerals use "
#         "subtractive notation for the values 4 (IV), 9 (IX), 40 (XL), 90 (XC), "
#         "400 (CD), and 900 (CM)."
#     ),
#     "user_id": "user_527",

#     "language": "python",
#     "verdict": "wrong_answer",

#     "source_code": """class Solution:
#     def intToRoman(self, num: int) -> str:
#         value_symbols = [
#             (1000, 'M'),
#             (500, 'D'),
#             (100, 'C'),
#             (50, 'L'),
#             (10, 'X'),
#             (5, 'V'),
#             (1, 'I')
#         ]

#         res = []

#         for value, symbol in value_symbols:
#             if num == 0:
#                 break

#             count = num // value
#             res.append(symbol * count)
#             num -= count * value

#         return ''.join(res)
# """,

#     "test_summary": {
#         "total_test_cases": 35,
#         "passed_test_cases": 29,
#         "failed_test_cases": 6,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": "num = 4",
#             "expected_output": "IV",
#             "actual_output": "IIII",
#         },
#         {
#             "stdin": "num = 9",
#             "expected_output": "IX",
#             "actual_output": "VIIII",
#         },
#         {
#             "stdin": "num = 944",
#             "expected_output": "CMXLIV",
#             "actual_output": "DCCCCXXXXIIII",
#         },
#     ],

#     "stdout": "DCCCCXXXXIIII",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 4,
#     "memory_kb": 11328,

#     "submitted_at": "2026-07-05T18:15:00Z",
# }

# VALID_WRONG_ANSWER_PAYLOAD = {
#     "submission_id": "sub_regex_001",
#     "problem_id": "regular_expression_matching",
#     "problem_statement": (
#         "Given an input string s and a pattern p, implement regular expression "
#         "matching with support for '.' and '*' where '.' matches any single "
#         "character and '*' matches zero or more of the preceding element. "
#         "Return a boolean indicating whether the matching covers the entire input string."
#     ),
#     "user_id": "user_742",

#     "language": "python",
#     "verdict": "wrong_answer",

#     "source_code": """class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         m = len(s)
#         n = len(p)

#         dp = [[False] * (n + 1) for _ in range(m + 1)]
#         dp[0][0] = True

#         for j in range(2, n + 1):
#             if p[j - 1] == '*':
#                 dp[0][j] = False          

#         for i in range(1, m + 1):
#             for j in range(1, n + 1):

#                 if p[j - 1] == '*':
#                     dp[i][j] = False     

#                     if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
#                         dp[i][j] = dp[i][j] or dp[i - 1][j]

#                 elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
#                     dp[i][j] = dp[i - 1][j - 1]

#         return dp[m][n]
# """,

#     "test_summary": {
#         "total_test_cases": 56,
#         "passed_test_cases": 48,
#         "failed_test_cases": 8,
#     },

#     "sample_failed_cases": [
#         {
#             "stdin": 's = "a"\np = "ab*"',
#             "expected_output": "true",
#             "actual_output": "false",
#         },
#         {
#             "stdin": 's = ""\np = "a*"',
#             "expected_output": "true",
#             "actual_output": "false",
#         },
#         {
#             "stdin": 's = "b"\np = "a*b"',
#             "expected_output": "true",
#             "actual_output": "false",
#         }
#     ],

#     "stdout": "false",
#     "stderr": "",
#     "compile_output": "",

#     "execution_time_ms": 10,
#     "memory_kb": 12320,

#     "submitted_at": "2026-07-05T22:10:00Z",
# }

VALID_WRONG_ANSWER_PAYLOAD = {
    "submission_id": "sub_subarraysum_006",
    "problem_id": "subarray_sum_equals_k",
    "problem_statement": (
        "Given an array of integers nums and an integer k, return the total "
        "number of contiguous subarrays whose sum equals k."
    ),
    "user_id": "user_675",
    "language": "python",
    "verdict": "wrong_answer",
    "source_code": """class Solution:
    def subarraySum(self, nums: list[int], k: int) -> int:
        count = 0
        prefix_sum = 0
        seen = {}
        for num in nums:
            prefix_sum += num
            if prefix_sum - k in seen:
                count += seen[prefix_sum - k]
            seen[prefix_sum] = seen.get(prefix_sum, 0) + 1
        return count
""",
    "test_summary": {
        "total_test_cases": 38,
        "passed_test_cases": 29,
        "failed_test_cases": 9,
    },
    "sample_failed_cases": [
        {
            "stdin": "nums = [1, 1, 1]\nk = 2",
            "expected_output": "2",
            "actual_output": "1",
        },
        {
            "stdin": "nums = [1, 2, 3]\nk = 3",
            "expected_output": "2",
            "actual_output": "1",
        },
    ],
    "stdout": "1",
    "stderr": "",
    "compile_output": "",
    "execution_time_ms": 7,
    "memory_kb": 11904,
    "submitted_at": "2026-07-06T13:05:00Z",
}