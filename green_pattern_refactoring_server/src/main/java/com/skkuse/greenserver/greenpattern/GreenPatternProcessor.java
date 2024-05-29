package com.skkuse.greenserver.greenpattern;

import com.skkuse.greenserver.code.Buggy;
import com.skkuse.greenserver.code.Fixed;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Base64;
import java.util.List;


@Component
@RequiredArgsConstructor
public class GreenPatternProcessor {
    private final List<CodeAnalyzer> codeAnalyzers;

    public Fixed getGreenPattern(Buggy buggy) {
        String buggyCode = buggy.base64ToJavaCode();
        for (CodeAnalyzer codeAnalyzer : codeAnalyzers) {
            buggyCode = codeAnalyzer.analyze(buggyCode);
        }
        return new Fixed(javaCodeToBase64(buggyCode));
    }

    private String javaCodeToBase64(String javaCode) {

        return Base64.getEncoder().encodeToString(javaCode.getBytes());
        // return Base64.getEncoder().encodeToString(javaCode.getBytes(StandardCharsets.UTF_8)); <- UTF8 인코딩 명시
    }
}
