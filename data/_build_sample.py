"""
Builds _sample_manifest.json from the hardcoded dataset in the document.
Run this once locally if you don't have the full 1000_manifest_final.json.
"""
import json, pathlib

SAMPLE = [
  {
    "problem_id": "f07r6rjgpeoc8h18g24o9ye9",
    "title": "Two Sum",
    "lc_id": "",
    "title_slug": "two-sum",
    "python_solution": "class Solution:\n    def twoSum(self, nums, target):\n        d = {}\n        for i, j in enumerate(nums):\n            r = target - j\n            if r in d: return [d[r], i]\n            d[j] = i",
    "python_solution_source": "cassanof_community",
    "python_solution_upvotes": 288,
    "python_solution_eri": "def twoSum(nums, target):\n    map = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in map:\n            return [map[complement], i]\n        map[num] = i\n    return []",
    "has_solution": True,
    "community_solutions": [
      {"code": "class Solution:\n    def twoSum(self, nums, target):\n        d = {}\n        for i, j in enumerate(nums):\n            r = target - j\n            if r in d: return [d[r], i]\n            d[j] = i", "upvotes": 288, "post_title": "Python Simple Solution || O(n) Time || O(n) Space"}
    ],
    "explanation_text": "The algorithm leverages a hash map. It iterates through the given 'nums' array and calculates the complementary value (target - current value). If the complementary value is already in the hash map, it means that we found a solution, and we return those indices. This approach has a time complexity of O(n) and a space complexity of O(n) as well.",
    "java_solution": "...",
    "cpp_solution": "...",
    "javascript_solution": "...",
    "full_markdown_solution": "### Approach 1: Brute Force\nLoop through each element x and find if there is another value that equals to target - x.\n### Approach 2: Two-pass Hash Table\nReduce the lookup time from O(n) to O(1) by trading space for speed.",
    "companies": ["Amazon","Google","Apple","Adobe","Microsoft","Bloomberg","Facebook","Oracle","Uber"],
    "asked_by_faang": True,
    "frequency": 1.0,
    "likes": 20217,
    "dislikes": 712,
    "rating": 0.97,
    "similar_questions": ["[3Sum","/problems/3sum/","Medium]","[4Sum","/problems/4sum/","Medium]","[Two Sum II - Input array is sorted","/problems/two-sum-ii-input-array-is-sorted/","Easy]"],
    "discuss_count": 999
  },
  {
    "problem_id": "it8cm2q8tth7amdc2lkfmspu",
    "title": "Add Two Numbers",
    "lc_id": "",
    "title_slug": "add-two-numbers",
    "python_solution": "class Solution:\n    def addTwoNumbers(self, l1, l2):\n        res = dummy = ListNode()\n        carry = 0\n        while l1 or l2:\n            v1, v2 = 0, 0\n            if l1: v1, l1 = l1.val, l1.next\n            if l2: v2, l2 = l2.val, l2.next\n            val = carry + v1 + v2\n            res.next = ListNode(val%10)\n            res, carry = res.next, val//10\n        if carry:\n            res.next = ListNode(carry)\n        return dummy.next",
    "python_solution_source": "cassanof_community",
    "python_solution_upvotes": 44,
    "python_solution_eri": "def addTwoNumbers(l1, l2):\n    dummy = ListNode(0)\n    current = dummy\n    carry = 0\n    while l1 or l2 or carry:\n        sum_val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry\n        carry = sum_val // 10\n        current.next = ListNode(sum_val % 10)\n        current = current.next\n        if l1: l1 = l1.next\n        if l2: l2 = l2.next\n    return dummy.next",
    "has_solution": True,
    "community_solutions": [],
    "explanation_text": "1. Initialize a dummy ListNode with a value of 0.\n2. Set current to that dummy ListNode, and set carry to 0.\n3. Iterate over the list nodes of l1 and l2, as well as the carry.\n4. Calculate the sum of the node values and carry, store the carry for the next iteration.\n5. Return the next of the dummy ListNode as a result.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...",
    "full_markdown_solution": "Keep track of the carry using a variable and simulate digits-by-digits sum.",
    "companies": ["Bloomberg","Microsoft","Amazon","Google","Facebook","Apple","Adobe","Paypal"],
    "asked_by_faang": True, "frequency": 0.931, "likes": 11350, "dislikes": 2704, "rating": 0.81,
    "similar_questions": ["[Multiply Strings","/problems/multiply-strings/","Medium]","[Add Binary","/problems/add-binary/","Easy]"],
    "discuss_count": 999
  },
  {
    "problem_id": "j0mk7wrgy4uvcl58bwmy92fl",
    "title": "Longest Substring Without Repeating Characters",
    "lc_id": "", "title_slug": "longest-substring-without-repeating-characters",
    "python_solution": "class Solution:\n    def lengthOfLongestSubstring(self, s):\n        if len(s) == 0: return 0\n        seen = {}\n        left, right = 0, 0\n        longest = 1\n        while right < len(s):\n            if s[right] in seen:\n                left = max(left,seen[s[right]]+1)\n            longest = max(longest, right - left + 1)\n            seen[s[right]] = right\n            right += 1\n        return longest",
    "python_solution_source": "cassanof_community", "python_solution_upvotes": 290,
    "python_solution_eri": "def length_of_longest_substring(s):\n    left = 0; right = 0; max_length = 0; characters = set()\n    while right < len(s):\n        if s[right] not in characters:\n            characters.add(s[right])\n            max_length = max(max_length, right - left + 1)\n            right += 1\n        else:\n            characters.remove(s[left])\n            left += 1\n    return max_length",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "The algorithm uses a sliding window with two pointers, left and right. It also uses a set to store unique characters in the current window. Initialize left and right pointers to the start of the string, and maxLength to 0. Move the right pointer and track seen characters. When a duplicate is found, move left forward. The algorithm runs in O(n) time.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Amazon","Bloomberg","Microsoft","Facebook","Apple","Adobe","eBay","Goldman Sachs","Google"],
    "asked_by_faang": True, "frequency": 0.909, "likes": 13810, "dislikes": 714, "rating": 0.95,
    "similar_questions": ["[Longest Substring with At Most Two Distinct Characters","/problems/longest-substring-with-at-most-two-distinct-characters/","Medium]"],
    "discuss_count": 999
  },
  {
    "problem_id": "yjtnksjtxxaot6kos627w55a",
    "title": "Median of Two Sorted Arrays",
    "lc_id": "", "title_slug": "median-of-two-sorted-arrays",
    "python_solution": None,
    "python_solution_source": "cassanof_community", "python_solution_upvotes": 32,
    "python_solution_eri": "def findMedianSortedArrays(nums1, nums2):\n    if len(nums1) > len(nums2):\n        return findMedianSortedArrays(nums2, nums1)\n    x, y = len(nums1), len(nums2)\n    low, high = 0, x\n    while low <= high:\n        partition_x = (low + high) // 2\n        partition_y = (x + y + 1) // 2 - partition_x\n        max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]\n        min_right_x = float('inf') if partition_x == x else nums1[partition_x]\n        max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]\n        min_right_y = float('inf') if partition_y == y else nums2[partition_y]\n        if max_left_x <= min_right_y and max_left_y <= min_right_x:\n            if (x + y) % 2 == 0:\n                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2\n            else:\n                return max(max_left_x, max_left_y)\n        elif max_left_x > min_right_y:\n            high = partition_x - 1\n        else:\n            low = partition_x + 1\n    return 0",
    "has_solution": True,
    "community_solutions": [
      {"code": "class Solution:\n    def findMedianSortedArrays(self, nums1, nums2):\n        return sorted(nums1+nums2)[(len(nums1)+len(nums2)-1)//2]", "upvotes": 32, "post_title": "Simple merge sort approach"}
    ],
    "explanation_text": "Use Binary Search. Choose the smaller array as nums1. Use Binary Search to partition the smallest array. Find four boundary values. If both maxLeftA<=minRightB and maxLeftB<=minRightA, partition is correct. Calculate median based on total length even/odd.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Amazon","Goldman Sachs","Facebook","Microsoft","Apple","Adobe","Google","Bloomberg"],
    "asked_by_faang": True, "frequency": 0.862, "likes": 9665, "dislikes": 1486, "rating": 0.87,
    "similar_questions": ["nan"],
    "discuss_count": 999
  },
  {
    "problem_id": "wco1t5htc04vz0601p97qfae",
    "title": "Longest Palindromic Substring",
    "lc_id": "", "title_slug": "longest-palindromic-substring",
    "python_solution": "class Solution:\n    def longestPalindrome(self, s):\n        n=len(s)\n        def expand(i,j):\n            while 0<=i<=j<n and s[i]==s[j]:\n                i-=1; j+=1\n            return (i+1, j)\n        res=(0,0)\n        for i in range(n):\n            b1 = expand(i,i)\n            b2 = expand(i,i+1)\n            res=max(res, b1, b2, key=lambda x: x[1]-x[0]+1)\n        return s[res[0]:res[1]]",
    "python_solution_source": "cassanof_community", "python_solution_upvotes": 47,
    "python_solution_eri": "def longest_palindromic_substring(s):\n    n = len(s)\n    if n == 0: return ''\n    start, max_length = 0, 1\n    for i in range(n):\n        l, r = i, i\n        while r < n - 1 and s[r] == s[r + 1]: r += 1\n        i = r\n        while l > 0 and r < n - 1 and s[l - 1] == s[r + 1]: l -= 1; r += 1\n        length = r - l + 1\n        if length > max_length: start, max_length = l, length\n    return s[start:start + max_length]",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "1. Initialize start and maxLength for result substring.\n2. Iterate through the given string s using the index i.\n3. For each index i, create two pointers l and r starting at i.\n4. Check if there's a consecutive sequence of identical characters.\n5. Expand the pointers l and r outwards to find the longest palindromic substring.\n6. Return the longest palindromic substring using the start and maxLength.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Amazon","Microsoft","Wayfair","Facebook","Adobe","eBay","Google","Oracle"],
    "asked_by_faang": True, "frequency": 0.847, "likes": 10271, "dislikes": 670, "rating": 0.94,
    "similar_questions": ["[Shortest Palindrome","/problems/shortest-palindrome/","Hard]","[Palindrome Pairs","/problems/palindrome-pairs/","Hard]"],
    "discuss_count": 999
  },
  {
    "problem_id": "eac2tw1ub8cj4javzbim6foh",
    "title": "Container With Most Water",
    "lc_id": "", "title_slug": "container-with-most-water",
    "python_solution": "class Solution:\n    def maxArea(self, height):\n        l, r, area = 0, len(height) - 1, 0\n        while l < r:\n            area = max(area, (r - l) * min(height[l], height[r]))\n            if height[l] < height[r]: l += 1\n            else: r -= 1\n        return area",
    "python_solution_source": "cassanof_community", "python_solution_upvotes": 133,
    "python_solution_eri": "def max_area(height):\n    max_area, left, right = 0, 0, len(height) - 1\n    while left < right:\n        max_area = max(max_area, min(height[left], height[right]) * (right - left))\n        if height[left] < height[right]: left += 1\n        else: right -= 1\n    return max_area",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "The algorithm uses a two-pointer approach. One pointer starts from the left end and the other from the right end. It calculates the area between these two lines and updates the maximum area. If the height at the left pointer is less than the height at the right pointer, it moves the left pointer to the right. Otherwise, it moves the right pointer to the left.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Amazon","Google","Microsoft","Facebook","Goldman Sachs","Adobe","Apple"],
    "asked_by_faang": True, "frequency": 0.673, "likes": 9031, "dislikes": 696, "rating": 0.93,
    "similar_questions": ["[Trapping Rain Water","/problems/trapping-rain-water/","Hard]"],
    "discuss_count": 999
  },
  {
    "problem_id": "vldye35kpp2whacwcp8i6u55",
    "title": "3Sum",
    "lc_id": "", "title_slug": "3sum",
    "python_solution": "def threeSum(nums):\n    nums.sort()\n    result = []\n    for i in range(len(nums) - 2):\n        if i == 0 or nums[i] != nums[i - 1]:\n            j, k = i + 1, len(nums) - 1\n            while j < k:\n                s = nums[i] + nums[j] + nums[k]\n                if s == 0:\n                    result.append([nums[i], nums[j], nums[k]])\n                    while j < k and nums[j] == nums[j + 1]: j += 1\n                    while j < k and nums[k] == nums[k - 1]: k -= 1\n                    j += 1; k -= 1\n                elif s < 0: j += 1\n                else: k -= 1\n    return result",
    "python_solution_source": "erichartford", "python_solution_upvotes": 0,
    "python_solution_eri": "def threeSum(nums):\n    nums.sort()\n    result = []\n    for i in range(len(nums) - 2):\n        if i == 0 or nums[i] != nums[i - 1]:\n            j, k = i + 1, len(nums) - 1\n            while j < k:\n                s = nums[i] + nums[j] + nums[k]\n                if s == 0:\n                    result.append([nums[i], nums[j], nums[k]])\n                    j += 1; k -= 1\n                elif s < 0: j += 1\n                else: k -= 1\n    return result",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "Sort the input array nums. Loop through nums with pointer i. For each i, initialize two pointers j and k. While j<k, calculate sum. If sum==0, add triplet to result and skip duplicates. Move pointers accordingly. Return result.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Amazon","Facebook","Microsoft","Bloomberg","Apple","Adobe","VMware","Google"],
    "asked_by_faang": True, "frequency": 0.788, "likes": 10032, "dislikes": 1035, "rating": 0.91,
    "similar_questions": ["[Two Sum","/problems/two-sum/","Easy]","[3Sum Closest","/problems/3sum-closest/","Medium]","[4Sum","/problems/4sum/","Medium]"],
    "discuss_count": 999
  },
  {
    "problem_id": "n9s2vg17dlxnobsrbqjz7lyo",
    "title": "Trapping Rain Water",
    "lc_id": "", "title_slug": "trapping-rain-water",
    "python_solution": "def trap(height):\n    n = len(height)\n    left, right, max_left, max_right, water = 0, n - 1, 0, 0, 0\n    while left < right:\n        if height[left] <= height[right]:\n            max_left = max(max_left, height[left])\n            water += max_left - height[left]\n            left += 1\n        else:\n            max_right = max(max_right, height[right])\n            water += max_right - height[right]\n            right -= 1\n    return water",
    "python_solution_source": "erichartford", "python_solution_upvotes": 0,
    "python_solution_eri": "def trap(height):\n    n = len(height)\n    left, right, max_left, max_right, water = 0, n - 1, 0, 0, 0\n    while left < right:\n        if height[left] <= height[right]:\n            max_left = max(max_left, height[left])\n            water += max_left - height[left]\n            left += 1\n        else:\n            max_right = max(max_right, height[right])\n            water += max_right - height[right]\n            right -= 1\n    return water",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "The algorithm uses a two-pointer approach, initializing left and right pointers to the beginning and end of the elevation map. It also initializes two variables maxLeft and maxRight to keep track of the maximum heights, and water to store the trapped water. The algorithm iterates until left < right. In each iteration, it compares the values at both pointers. If left value is less than or equal to right value, water can be trapped on the left side. This algorithm has O(n) time and O(1) space.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Goldman Sachs","Facebook","Amazon","Microsoft","Bloomberg","Apple","Databricks","Google"],
    "asked_by_faang": True, "frequency": 0.963, "likes": 10683, "dislikes": 159, "rating": 0.99,
    "similar_questions": ["[Container With Most Water","/problems/container-with-most-water/","Medium]","[Product of Array Except Self","/problems/product-of-array-except-self/","Medium]"],
    "discuss_count": 999
  },
  {
    "problem_id": "xknj233zz4mx1krwjb8ov16u",
    "title": "Valid Parentheses",
    "lc_id": "", "title_slug": "valid-parentheses",
    "python_solution": "def is_valid(s):\n    stack = []\n    for c in s:\n        if c in '([{': stack.append(c)\n        else:\n            if not stack: return False\n            if c == ')' and stack[-1] != '(': return False\n            if c == '}' and stack[-1] != '{': return False\n            if c == ']' and stack[-1] != '[': return False\n            stack.pop()\n    return not stack",
    "python_solution_source": "erichartford", "python_solution_upvotes": 0,
    "python_solution_eri": "def is_valid(s):\n    stack = []\n    for c in s:\n        if c in '([{': stack.append(c)\n        else:\n            if not stack: return False\n            if c == ')' and stack[-1] != '(': return False\n            if c == '}' and stack[-1] != '{': return False\n            if c == ']' and stack[-1] != '[': return False\n            stack.pop()\n    return not stack",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "The algorithm uses a stack data structure. It iterates through the string one character at a time. When an open bracket is encountered, it is pushed onto the stack. When a close bracket is encountered, the algorithm checks if the stack is empty or the corresponding open bracket is not at the top of the stack. If either condition is true, the function returns false. If not, the open bracket is popped from the stack. After iterating, if the stack is not empty, there were unmatched open braces, so return false. Otherwise, return true.",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Amazon","Bloomberg","Facebook","Apple","Microsoft","Expedia","Spotify","Google"],
    "asked_by_faang": True, "frequency": 0.902, "likes": 7188, "dislikes": 294, "rating": 0.96,
    "similar_questions": ["[Generate Parentheses","/problems/generate-parentheses/","Medium]","[Longest Valid Parentheses","/problems/longest-valid-parentheses/","Hard]"],
    "discuss_count": 999
  },
  {
    "problem_id": "rmnqrwq1u98mq9jofjtgj9h2",
    "title": "Maximum Subarray",
    "lc_id": "", "title_slug": "maximum-subarray",
    "python_solution": "def maxSubArray(nums):\n    max_sum = current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(current_sum + num, num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum",
    "python_solution_source": "erichartford", "python_solution_upvotes": 0,
    "python_solution_eri": "def maxSubArray(nums):\n    max_sum = current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(current_sum + num, num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum",
    "has_solution": True, "community_solutions": [],
    "explanation_text": "The algorithm uses Kadane's Algorithm. It iterates through the given array once and tracks the maximum sum found so far and the current sum. For each element, we compare the sum of the current_sum with the element itself, and select the maximum. This helps decide whether to continue the current contiguous subarray or start a new one. Then, we compare the new current_sum with our global max_sum, and update max_sum if larger. Time complexity: O(n), Space complexity: O(1).",
    "java_solution": "...", "cpp_solution": "...", "javascript_solution": "...", "full_markdown_solution": "",
    "companies": ["Microsoft","Amazon","Apple","LinkedIn","ByteDance","Google","Adobe","Cisco","Facebook"],
    "asked_by_faang": True, "frequency": 0.802, "likes": 11458, "dislikes": 551, "rating": 0.95,
    "similar_questions": ["[Best Time to Buy and Sell Stock","/problems/best-time-to-buy-and-sell-stock/","Easy]","[Maximum Product Subarray","/problems/maximum-product-subarray/","Medium]"],
    "discuss_count": 999
  }
]

out = pathlib.Path(__file__).parent / "_sample_manifest.json"
out.write_text(json.dumps(SAMPLE, indent=2))
print(f"Written {len(SAMPLE)} problems to {out}")
