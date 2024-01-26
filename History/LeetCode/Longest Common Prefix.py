class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        j = 0
        pfx = ''
        try:
            while True:

                for i in range(len(strs)):
                    first_change = strs[i][j]

                    if len(strs) == 1:
                        return pfx

                    if i > 0 and first_change != last_value:
                        return pfx

                    if i == len(strs) - 1 and first_change == last_value:
                        pfx += first_change

                    last_value = strs[i][j]

                j += 1

        except IndexError:
            return pfx


test = Solution()
print(test.longestCommonPrefix(['a']))
