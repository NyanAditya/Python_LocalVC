class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        i, j = 0, 1
        while True:

            suml = nums[i] + nums[j]

            if suml == target:
                return [i, j]

            elif j == len(nums) - 1:
                i += 1
                j = i + 1

            else:
                j += 1


obj = Solution()
print(obj.twoSum([3,2,4], 6))
