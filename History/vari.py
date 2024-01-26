class Solution(object):
    def twoSum(self, nums, target):
        table = {}

        for i in range(len(nums)):
            if nums[i] not in table:
                table[(target-nums[i])] = i
            else:
                return [i, table[nums[i]]]


obj = Solution()
print(obj.twoSum([3,2,4], 6))
