package com.skkuse.greenserver.greenpattern;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class IterateNested implements CodeAnalyzer {
    @Override
    public String analyze(String buggyCode) {
        boolean isDetected = false;
        int classStartIndex = 0;
        int firstStartIfIndex = 0;
        int firstEndIfIndex = 0;
        int thirdStartIfIndex = 0;
        int thirdEndIfIndex = 0;
        String firstCondition = "";
        String secondCondition = "";
        String thirdCondition = "";

        StringBuilder fixedCodeBuilder = new StringBuilder();

        try {
            // 코드 분할
            String[] codes = buggyCode.split("\n");
            ArrayList<String> lines = new ArrayList<>(Arrays.asList(codes));

            // 검출
            int lineSize = lines.size();
            boolean nestedIfFound = false;
            Pattern pattern = Pattern.compile("\\((.*?)\\)");

            for(int i=0; i<lineSize; i++) {
                String line = codes[i];
                if (line.contains("public class Buggy")) {
                    classStartIndex = i;
                    continue;
                }

                if (line.contains("if(")) {
                    Matcher matcher1 = pattern.matcher(line);
                    firstStartIfIndex = i;

                    if(matcher1.find()) {
                        firstCondition = matcher1.group(1);
                    }

                    for (int j = i+1; j < lineSize; j++) {
                        String line2 = lines.get(j);

                        if(line2.contains("if(")) {
                            Matcher matcher2 = pattern.matcher(line2);

                            if(matcher2.find()) {
                                secondCondition = matcher2.group(1);
                            }

                            for(int k = j+1; k < lineSize; k++) {
                                String line3 = lines.get(k);

                                if(line3.contains("if(")) {
                                    nestedIfFound = true;
                                    thirdStartIfIndex = k;
                                    Matcher matcher3 = pattern.matcher(line3);

                                    if(matcher3.find()) {
                                        thirdCondition = matcher3.group(1);
                                    }

                                    for(int l = k+1; l < lineSize; l++) {
                                        String line4 = lines.get(l);
                                        if(line4.contains("}")) {
                                            thirdEndIfIndex = l;
                                            break;
                                        }
                                    }

                                    int count = 0;

                                    for(int l = k+1; l < lineSize; l++) {
                                        String line4 = lines.get(l);

                                        if(line4.contains("}")) {
                                            count++;
                                            if(count==3) {
                                                firstEndIfIndex = l;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if(thirdEndIfIndex != 0) break;
                            }
                        }
                        if(thirdEndIfIndex != 0) break;
                    }

                    if (nestedIfFound) {
                        break;
                    }
                }
            }

            // 수정
            if (nestedIfFound) {
                lines.set(classStartIndex, "public class Fixed {\n");
                String conditionBody = "";
                for(int i=thirdStartIfIndex+1; i<thirdEndIfIndex; i++) {
                    conditionBody = conditionBody + (lines.get(i) + "\n");
                }

                for(int i=firstStartIfIndex; i<=firstEndIfIndex; i++) {
                    lines.set(i, "##MUSTDELETE##");
                }

                lines.set(firstStartIfIndex-1, "\t\tif((" + firstCondition + " && " + secondCondition + ") && " + thirdCondition + ") {\n");
                lines.add(firstStartIfIndex, "\t\t\t" + conditionBody + "\n");
                lines.add(firstStartIfIndex+1, "\t\t}\n");

                lines.removeIf(item -> item.equals("##MUSTDELETE##"));

                // 수정된 코드를 문자열로 변환
                for (String line : lines) {
                    fixedCodeBuilder.append(line).append("\n");
                }
            } else {
                // 중첩된 if문이 없는 경우 원본 코드를 그대로 반환
                fixedCodeBuilder.append(buggyCode);
            }
        } catch(Exception e) {
            e.printStackTrace();
        }

        return fixedCodeBuilder.toString();
    }
}
