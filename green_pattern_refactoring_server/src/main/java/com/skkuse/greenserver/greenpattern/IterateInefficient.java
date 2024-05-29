package com.skkuse.greenserver.greenpattern;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
public class IterateInefficient implements CodeAnalyzer{

    @Override
    public String analyze(String buggyCode) {
        StringBuilder fixedCodeBuilder = new StringBuilder();
        boolean isDetected = false;
        int classStartIndex = 0;

        try {
            // 코드 분할
            String[] codes = buggyCode.split("\n");
            ArrayList<String> lines = new ArrayList<>(Arrays.asList(codes));

            // 검출
            int buggyLine = -1;
            String arrayListVariableName = "";
            int lineSize = lines.size();

            for(int i=0; i<lineSize; i++) {
                String line = codes[i];
                if(line.contains("public class Buggy")) {
                    classStartIndex = i;
                    continue;
                }

                Pattern pattern = Pattern.compile("\\bArrayList\\s*<[^>]*>\\s+([a-zA-Z0-9_]+)\\s*=");

                Matcher matcher = pattern.matcher(line);
                if (matcher.find()) {
                    arrayListVariableName = matcher.group(1);

                    for(int j=i; j<codes.length; j++) {
                        String line2 = codes[j];
                        Pattern sizeMethodPattern = Pattern.compile("\\b" + arrayListVariableName + "\\.size\\(\\)");

                        Matcher sizeMethodMatcher = sizeMethodPattern.matcher(line2);
                        if (sizeMethodMatcher.find()) {
//                            System.out.println("ArrayList 객체 변수명: " + arrayListVariableName);
//                            System.out.println("해당 변수를 사용하여 size() 메서드를 호출함");
                            buggyLine = j;
                            isDetected = true;
                            break;
                        }
                    }
                }
            }

            // 수정
            if(isDetected) {
                int count = countLeadingSpaces(lines.get(buggyLine));
                lines.set(classStartIndex, "public class Fixed {");

                lines.set(buggyLine, lines.get(buggyLine).replace(arrayListVariableName + ".size()", arrayListVariableName + "Size"));

                StringBuilder builder = new StringBuilder();
                for(int i=0; i<count; i++) {
                    builder.append(' ');
                }
                builder.append("int " + arrayListVariableName + "Size = " + arrayListVariableName + ".size();\n");
                lines.add(buggyLine, builder.toString());
            }

            // 수정된 코드를 문자열로 변환
            for (String line : lines) {
                fixedCodeBuilder.append(line).append("\n");
            }
        } catch (Exception e) {
            System.out.println(e);
        }
        return fixedCodeBuilder.toString();
    }

    private static int countLeadingSpaces(String text) {
        int count = 0;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == ' ') {
                count++;
            } else {
                break;
            }
        }
        return count;
    }
}
