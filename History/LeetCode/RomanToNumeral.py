class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        vdic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        # vdic2 = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
        total = 0
        last_view = 9999

        for i in s:
            if last_view < vdic[i]:
                if vdic[i] - last_view == 4:
                    total += 3
                    continue

                elif vdic[i] - last_view == 9:
                    total += 8
                    continue

                elif vdic[i] - last_view == 40:
                    total += 30
                    continue

                elif vdic[i] - last_view == 90:
                    total += 80
                    continue

                elif vdic[i] - last_view == 400:
                    total += 300
                    continue

                elif vdic[i] - last_view == 900:
                    total += 800
                    continue

            total += vdic[i]
            last_view = vdic[i]
        return total


inp = Solution()
print(inp.romanToInt('MDXC'))
